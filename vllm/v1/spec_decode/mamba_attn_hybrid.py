# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from dataclasses import replace
from typing import Any, cast

import torch
from torch import nn
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel

logger = init_logger(__name__)


class MambaAttnHybridProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "mamba_attn_hybrid"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )

        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        self.parallel_drafting_hidden_state_tensor = None

        self._draft_attn_layers: list | None = None
        self._verifier_attns: list | None = None
        self._kv_sharing_active: bool = False

        self._latent_fusion_F: int | None = None
        self._latent_seed_buffer: torch.Tensor | None = None

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        return replace(
            base,
            attention_config=replace(
                base.attention_config,
                use_non_causal=True,
            ),
        )

    @override
    def _warn_if_multimodal(self):
        pass

    @override
    def _parallel_sample_draft(
        self,
        sample_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        sampling_metadata,
    ) -> torch.Tensor:
        if not self.model.has_markov_head:
            return super()._parallel_sample_draft(
                sample_hidden_states, next_token_ids, sampling_metadata
            )
        greedy = (
            not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy
        )
        if not greedy:
            raise NotImplementedError(
                "DSpark (mamba_attn_hybrid + markov_rank>0) drafting supports only "
                "greedy sampling; the reference's seeded-Gumbel probabilistic path "
                "has no equivalent in this fork. Run with greedy sampling."
            )
        n_spec = self.num_speculative_tokens
        num_reqs = next_token_ids.shape[0]
        base_logits = self.model.compute_draft_logits(sample_hidden_states)
        draft_vocab = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, draft_vocab)

        draft_tokens = torch.empty(
            (num_reqs, n_spec), dtype=torch.int64, device=base_logits.device
        )
        prev = next_token_ids.to(torch.long)
        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            tok = self.model.map_draft_to_target(logits_i.argmax(dim=-1))
            draft_tokens[:, i] = tok
            prev = tok
        return draft_tokens

    @override
    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)

        from vllm.model_executor.models.interfaces import (
            SupportsMultiModal,
            supports_multimodal,
        )

        if supports_multimodal(target_model):
            target_language_model = cast(
                SupportsMultiModal, target_model
            ).get_language_model()
        else:
            target_language_model = target_model

        hf_config = self.draft_model_config.hf_config

        assert getattr(hf_config, "layer_mapping", None) is None, (
            "Mamba-Attn-Hybrid is identity-only; layer_mapping must be None."
        )

        self._draft_attn_layers = [
            sub.self_attn.attn for sub in self.model.model.attn_sublayers
        ]
        num_attn_layers = len(self._draft_attn_layers)

        verifier_kv_layer_ids = getattr(hf_config, "attn_kv_layer_ids", None)
        assert verifier_kv_layer_ids is not None, (
            "Checkpoint must have attn_kv_layer_ids in config (one verifier KV "
            "layer per attention sub-layer, in attention-block order)."
        )
        assert len(verifier_kv_layer_ids) == num_attn_layers, (
            f"attn_kv_layer_ids has {len(verifier_kv_layer_ids)} entries but the "
            f"recipe has {num_attn_layers} attention sub-layer(s); they must match."
        )

        latent_fusion_layer_ids = getattr(
            hf_config, "latent_fusion_layer_ids", None
        )
        if latent_fusion_layer_ids is None:
            latent_fusion_layer_ids = getattr(
                hf_config, "eagle_aux_hidden_state_layer_ids", None
            )
        assert latent_fusion_layer_ids is not None, (
            "Checkpoint must have latent_fusion_layer_ids or "
            "eagle_aux_hidden_state_layer_ids to size the latent seed."
        )
        self._latent_fusion_F = len(latent_fusion_layer_ids)

        draft_hidden_size = hf_config.hidden_size
        self._latent_seed_buffer = torch.zeros(
            self.max_batch_size,
            self._latent_fusion_F * draft_hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

        assert self._latent_fusion_F == self.model.model.num_fusion_layers, (
            f"latent fusion F mismatch: proposer={self._latent_fusion_F} "
            f"model={self.model.model.num_fusion_layers}"
        )

        all_verifier_attns = [
            layer.self_attn.attn for layer in target_language_model.model.layers
        ]
        n_verifier_layers = len(all_verifier_attns)
        bad_kv = [
            lid for lid in verifier_kv_layer_ids if not 0 <= lid < n_verifier_layers
        ]
        if bad_kv:
            raise ValueError(
                f"attn_kv_layer_ids {bad_kv} out of range for a verifier with "
                f"{n_verifier_layers} layers (valid 0..{n_verifier_layers - 1}); "
                "keep --attn-kv-layer-ids consistent with the verifier."
            )
        self._verifier_attns = [
            all_verifier_attns[lid] for lid in verifier_kv_layer_ids
        ]

        if self.speculative_config.enable_kv_sharing:
            for draft_attn, verifier_lid in zip(
                self._draft_attn_layers, verifier_kv_layer_ids
            ):
                draft_attn.kv_sharing_target_layer_name = (
                    f"model.layers.{verifier_lid}.self_attn.attn"
                )
            self._kv_sharing_active = True

        logger.info(
            "MambaAttnHybrid: %d attention sub-layers, verifier KV layers %s, "
            "F=%d fusion layers %s, kv_sharing=%s",
            num_attn_layers,
            list(verifier_kv_layer_ids),
            self._latent_fusion_F,
            list(latent_fusion_layer_ids),
            self._kv_sharing_active,
        )

    def _copy_verifier_kv_to_draft(
        self,
        context_slot_mapping: torch.Tensor,
    ) -> None:
        assert self._draft_attn_layers is not None
        assert self._verifier_attns is not None
        for draft_attn, verifier_attn in zip(
            self._draft_attn_layers, self._verifier_attns
        ):
            if draft_attn.kv_sharing_target_layer_name is not None:
                continue
            verifier_kv = verifier_attn.kv_cache
            cache_block_size = verifier_kv.shape[2]
            block_ids = context_slot_mapping // cache_block_size
            offsets = context_slot_mapping % cache_block_size
            k = verifier_kv[0, block_ids, offsets]
            v = verifier_kv[1, block_ids, offsets]
            draft_attn.impl.do_kv_cache_update(
                draft_attn, k, v, draft_attn.kv_cache, context_slot_mapping,
            )

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        self._dflash_num_context = num_context

        self._dflash_hidden_states = target_hidden_states

        anchor_indices = cad.query_start_loc[1:] - 1
        if num_rejected_tokens_gpu is not None:
            anchor_indices = anchor_indices - num_rejected_tokens_gpu
        self._dflash_anchor_indices = anchor_indices

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        copy_and_expand_dflash_inputs_kernel[grid](
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        new_seq_lens_cpu_upper_bound = (
            cad.seq_lens_cpu_upper_bound + num_query_per_req
            if cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=query_slot_mapping,
            causal=False,
        )

        return num_query_total, token_indices_to_sample, new_cad

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}


        if self._latent_seed_buffer is not None:
            self._latent_seed_buffer.zero_()

        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
                latent_seed=self._latent_seed_buffer,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:

        if not self._kv_sharing_active:
            self._copy_verifier_kv_to_draft(
                self._context_slot_mapping_buffer[: self._dflash_num_context],
            )

        assert self._latent_seed_buffer is not None
        anchor_rows = self._dflash_hidden_states[self._dflash_anchor_indices]
        num_reqs = anchor_rows.shape[0]
        self._latent_seed_buffer[:num_reqs].copy_(anchor_rows)

        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
                latent_seed=self._latent_seed_buffer,
            ),
            num_input_tokens,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(
            cad, draft_index
        )

        sliding_layer_names: set[str] = getattr(
            self.model, "sliding_attention_layer_names", set()
        )
        if sliding_layer_names:
            for attn_group in self.draft_attn_groups:
                causal_layers = sliding_layer_names & set(attn_group.layer_names)
                if causal_layers:
                    causal_attn_metadata = (
                        attn_group.get_metadata_builder().build_for_drafting(
                            common_attn_metadata=cad.replace(causal=True),
                            draft_index=draft_index,
                        )
                    )
                    for layer_name in causal_layers:
                        per_layer[layer_name] = causal_attn_metadata

        for layer_name, attn_metadata in per_layer.items():
            if layer_name in sliding_layer_names:
                assert getattr(attn_metadata, "causal", None) is True, (
                    f"Attention metadata for sliding layer {layer_name} does not have"
                    " causal support, which is required for DFlash SWA."
                )
                continue
            assert getattr(attn_metadata, "causal", None) is False, (
                f"Attention metadata for layer {layer_name} is not non-causal,"
                " which is required for Mamba-Attn-Hybrid full-attention layers."
            )
        return per_group, per_layer

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        use_aux_hidden_state = True
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state
