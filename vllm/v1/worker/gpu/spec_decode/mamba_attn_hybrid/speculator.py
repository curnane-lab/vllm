# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DPSyncState
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.mamba_attn_hybrid.utils import (
    load_mamba_attn_hybrid_model,
    wire_zero_copy_kv,
)

logger = init_logger(__name__)


class MambaAttnHybridSpeculator(DSparkSpeculator):
    _speculator_name = "MambaAttnHybrid"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        hf_config = self.draft_model_config.hf_config

        self.sample_from_anchor = False
        self.num_query_per_req = 1 + self.num_speculative_steps
        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        fusion_layer_ids = getattr(hf_config, "latent_fusion_layer_ids", None) or getattr(
            hf_config, "eagle_aux_hidden_state_layer_ids", None
        )
        if not fusion_layer_ids:
            raise ValueError(
                "mamba_attn_hybrid needs latent_fusion_layer_ids (or "
                "eagle_aux_hidden_state_layer_ids) to size the latent seed."
            )
        self.num_fusion_layers = len(fusion_layer_ids)
        hidden = self.draft_model_config.get_hidden_size()

        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            self.num_fusion_layers * hidden,
            dtype=self.dtype,
            device=device,
        )
        self.latent_seed = torch.zeros(
            self.max_num_reqs,
            self.num_fusion_layers * hidden,
            dtype=self.dtype,
            device=device,
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_mamba_attn_hybrid_model(target_model, self.vllm_config)
        kv_ids = wire_zero_copy_kv(model, target_model)

        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = torch.arange(d2t.shape[0], device=d2t.device) + d2t
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
            logger.info(
                "MambaAttnHybrid: reduced draft vocab (%d -> %d); probabilistic "
                "scatter map built",
                d2t.shape[0],
                self.vocab_size,
            )

        logger.info(
            "MambaAttnHybrid: %d attention sub-layers <- verifier KV %s, F=%d "
            "fusion layers, zero-copy KV sharing enabled, draft sampling=%s",
            len(kv_ids),
            kv_ids,
            self.num_fusion_layers,
            "sequential Markov"
            if getattr(model, "has_markov_head", False)
            else "parallel",
        )
        return model

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers=None,
        target_attn_groups=None,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        sliding = getattr(self.model, "sliding_attention_layer_names", set())
        if sliding and len(sliding) != len(self.model.model.attn_sublayers):
            raise NotImplementedError(
                "V2 builds one attention metadata for the whole draft, so a "
                "recipe mixing sliding and full attention is not supported yet "
                f"({len(sliding)} of {len(self.model.model.attn_sublayers)} "
                "sub-layers are sliding). Train with --sliding-window-indices "
                "covering all attention blocks, or none."
            )
        self.dflash_causal = bool(sliding)

    def capture(self, attn_states: dict | None = None) -> None:
        self.latent_seed.zero_()
        assert self.query_cudagraph_manager is not None
        self.query_cudagraph_manager.causal = self.dflash_causal
        super().capture(attn_states)

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        from vllm.forward_context import BatchDescriptor, set_forward_context

        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            dp_sync=dp_sync,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            return self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
                latent_seed=self.latent_seed,
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        if getattr(self.model, "has_markov_head", False):
            super()._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        else:
            DFlashSpeculator._generate_draft(
                self,
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        if aux_hidden_states:
            num_reqs = input_batch.num_reqs
            fused = torch.cat(aux_hidden_states, dim=-1)
            anchor = (
                input_batch.query_start_loc[1 : num_reqs + 1]
                - 1
                - num_rejected[:num_reqs]
            )
            self.latent_seed[:num_reqs].copy_(fused[anchor])
        else:
            last_hidden_states = last_hidden_states.new_zeros(
                last_hidden_states.shape[0], self.hidden_states.shape[1]
            )

        return super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp=num_tokens_across_dp,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
        )
