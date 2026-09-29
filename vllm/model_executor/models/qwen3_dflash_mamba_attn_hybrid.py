# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Iterable

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM

from .qwen3_dflash_hybrid_blocks import (
    DFlashQwen3Attention,
    MambaDraftMixer,
    _get_dflash_layer_types,
)
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


class HybridMambaSubLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        d_state: int = 16,
        num_heads: int = 64,
        head_dim: int = 64,
        n_groups: int = 1,
        conv_kernel: int = 4,
        expand: int = 1,
    ) -> None:
        super().__init__()
        self.ssm = MambaDraftMixer(
            hidden_size=config.hidden_size,
            d_state=d_state,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            expand=expand,
        )
        self.ssm_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_reqs: int,
        ssm_state0: torch.Tensor | None,
    ) -> torch.Tensor:
        AK, d = hidden_states.shape
        K = AK // num_reqs
        h = hidden_states.reshape(num_reqs, K, d)
        h = h + self.ssm(self.ssm_norm(h), ssm_state0)
        return h.reshape(AK, d)


class HybridAttnSubLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        layer_type: str = "full_attention",
        attn_rope_entry: dict | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        rope_parameters = (
            attn_rope_entry["rope_parameters"]
            if attn_rope_entry is not None
            else config.rope_parameters
        )
        draft_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )
        verifier_window = (
            attn_rope_entry["sliding_window"] if attn_rope_entry is not None else None
        )
        sliding_window = draft_window if draft_window is not None else verifier_window
        self.layer_type = (
            "sliding_attention" if sliding_window is not None else "full_attention"
        )
        self.self_attn = DFlashQwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            sliding_window=sliding_window,
            rope_parameters=rope_parameters,
            qk_norm=getattr(config, "use_qk_norm", True),
            prefix=f"{prefix}.self_attn",
            attn_type=AttentionType.DECODER,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        return residual + hidden_states


class HybridMLPSubLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class DSparkMarkovHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w1")
        )
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w2")
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:
        return logits_processor(self.markov_w2, markov_embed)


@support_torch_compile
class MambaAttnHybridQwen3Model(nn.Module):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        if drafter_config is not None and "use_aux_hidden_state" in drafter_config:
            self.use_aux_hidden_state = drafter_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        if getattr(self.config, "attn_rope", None) is None:
            set_default_rope_theta(self.config, default_theta=1000000)

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        block_pattern = list(getattr(self.config, "block_pattern", None) or [])
        if not block_pattern:
            raise ValueError(
                "Mamba-Attn-Hybrid requires a non-empty block_pattern in config."
            )
        self.block_pattern = block_pattern
        attn_kv_layer_ids = list(getattr(self.config, "attn_kv_layer_ids", None) or [])
        num_attn = block_pattern.count("attention")
        num_mamba = block_pattern.count("mamba")
        if len(attn_kv_layer_ids) != num_attn:
            raise ValueError(
                f"attn_kv_layer_ids has {len(attn_kv_layer_ids)} entries but "
                f"block_pattern has {num_attn} attention sub-layer(s); must match."
            )
        self.attn_kv_layer_ids = attn_kv_layer_ids

        self.attn_rope = getattr(self.config, "attn_rope", None)

        if self.config.num_hidden_layers != num_attn:
            raise ValueError(
                f"num_hidden_layers ({self.config.num_hidden_layers}) must equal the "
                f"number of attention sub-layers ({num_attn}); train.py derives "
                "--num-layers from --block-pattern."
            )
        self.layer_types = _get_dflash_layer_types(self.config)

        if getattr(self.config, "sliding_window_non_causal", False):
            raise NotImplementedError(
                "mamba_attn_hybrid inference does not support "
                "sliding_window_non_causal; retrain without it or extend the "
                "proposer's SWA attention metadata."
            )
        if getattr(self.config, "sliding_window_prefix", 0):
            raise NotImplementedError(
                "mamba_attn_hybrid inference does not support sliding_window_prefix "
                "(attention sink); retrain without it or extend the attention mask."
            )

        self.mamba_d_state = getattr(self.config, "mamba_d_state", 16)
        self.mamba_num_heads = getattr(self.config, "mamba_num_heads", 64)
        self.mamba_head_dim = getattr(self.config, "mamba_head_dim", 64)
        self.mamba_n_groups = getattr(self.config, "mamba_n_groups", 1)
        self.mamba_conv_kernel = getattr(self.config, "mamba_conv_kernel", 4)
        self.mamba_expand = getattr(self.config, "mamba_expand", 1)
        self.mamba_seed_mode = getattr(self.config, "mamba_seed_mode", "per_layer")

        self.block_size = 1 + vllm_config.speculative_config.num_speculative_tokens

        layers: list[nn.Module] = []
        self._sub_type: list[str] = []
        self._sub_seed_idx: list[int] = []
        attn_i = 0
        mamba_i = 0
        for flat_idx, tok in enumerate(block_pattern):
            sub_prefix = maybe_prefix(prefix, f"layers.{flat_idx + start_layer_id}")
            if tok == "mamba":
                layers.append(
                    HybridMambaSubLayer(
                        self.config,
                        d_state=self.mamba_d_state,
                        num_heads=self.mamba_num_heads,
                        head_dim=self.mamba_head_dim,
                        n_groups=self.mamba_n_groups,
                        conv_kernel=self.mamba_conv_kernel,
                        expand=self.mamba_expand,
                    )
                )
                self._sub_type.append("mamba")
                if self.mamba_seed_mode == "per_layer":
                    self._sub_seed_idx.append(mamba_i)
                elif self.mamba_seed_mode == "shared":
                    self._sub_seed_idx.append(0)
                else:
                    self._sub_seed_idx.append(0 if mamba_i == 0 else -1)
                mamba_i += 1
            elif tok == "attention":
                attn_rope_entry = (
                    self.attn_rope[attn_i] if self.attn_rope is not None else None
                )
                layers.append(
                    HybridAttnSubLayer(
                        self.config,
                        cache_config=current_vllm_config.cache_config,
                        quant_config=self.quant_config,
                        layer_type=self.layer_types[attn_i],
                        attn_rope_entry=attn_rope_entry,
                        prefix=sub_prefix,
                    )
                )
                self._sub_type.append("attention")
                self._sub_seed_idx.append(-1)
                attn_i += 1
            else:
                layers.append(
                    HybridMLPSubLayer(
                        self.config,
                        quant_config=self.quant_config,
                        prefix=sub_prefix,
                    )
                )
                self._sub_type.append("mlp")
                self._sub_seed_idx.append(-1)
        self.layers = nn.ModuleList(layers)

        self.attn_sublayers: list[HybridAttnSubLayer] = [
            layer
            for layer, t in zip(self.layers, self._sub_type)
            if t == "attention"
        ]
        self.sliding_attention_layer_names = {
            sub.self_attn.attn.layer_name
            for sub in self.attn_sublayers
            if sub.layer_type == "sliding_attention"
        }

        latent_fusion_layer_ids = getattr(
            self.config, "latent_fusion_layer_ids", None
        )
        if latent_fusion_layer_ids is None:
            latent_fusion_layer_ids = getattr(
                self.config, "eagle_aux_hidden_state_layer_ids", None
            )
        if latent_fusion_layer_ids is None:
            raise ValueError(
                "Mamba-Attn-Hybrid requires latent_fusion_layer_ids (or "
                "eagle_aux_hidden_state_layer_ids) to size the fc projection."
            )
        self.num_fusion_layers = len(latent_fusion_layer_ids)
        self.fc = nn.Linear(
            self.num_fusion_layers * self.config.hidden_size,
            self.config.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.fc_norm = (
            nn.ModuleList(
                [
                    RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
                    for _ in range(self.num_fusion_layers)
                ]
            )
            if getattr(self.config, "fc_norm", False)
            else None
        )

        seed_out = self.mamba_num_heads * self.mamba_head_dim * self.mamba_d_state
        num_seed_projs = num_mamba if self.mamba_seed_mode == "per_layer" else 1
        self.seed_projs = nn.ModuleList(
            [
                nn.Linear(self.config.hidden_size, seed_out, bias=False)
                for _ in range(num_seed_projs)
            ]
        )

        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        markov_rank = getattr(self.config, "markov_rank", 0) or 0
        if markov_rank > 0:
            markov_head_type = getattr(self.config, "markov_head_type", "vanilla")
            if markov_head_type != "vanilla":
                raise ValueError(
                    "DSpark mamba_attn_hybrid inference supports only "
                    f"markov_head_type='vanilla', got {markov_head_type!r}."
                )
            draft_vocab_size = (
                getattr(self.config, "draft_vocab_size", None) or self.config.vocab_size
            )
            self.markov_head: DSparkMarkovHead | None = DSparkMarkovHead(
                self.config.vocab_size,
                draft_vocab_size,
                markov_rank,
                prefix=maybe_prefix(prefix, "markov_head"),
            )
        else:
            self.markov_head = None

        logger.info(
            "Mamba-Attn-Hybrid: recipe %s | %d attention sub-layers <- verifier KV %s "
            "| F=%d fusion | seed_mode=%s",
            " ".join(block_pattern),
            num_attn,
            list(attn_kv_layer_ids),
            self.num_fusion_layers,
            self.mamba_seed_mode,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        latent_seed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        assert latent_seed is not None, (
            "MambaAttnHybridQwen3Model.forward requires latent_seed to be passed "
            "by the proposer (as a forward argument, not an attribute)."
        )
        num_reqs = hidden_states.shape[0] // self.block_size
        latent = latent_seed[:num_reqs]
        if self.fc_norm is not None:
            chunks = latent.chunk(len(self.fc_norm), dim=-1)
            latent = torch.cat(
                [norm(c) for norm, c in zip(self.fc_norm, chunks, strict=True)], dim=-1
            )
        z = self.hidden_norm(self.fc(latent))

        seed_states = [
            proj(z).view(
                num_reqs,
                self.mamba_num_heads,
                self.mamba_head_dim,
                self.mamba_d_state,
            )
            for proj in self.seed_projs
        ]
        if self.mamba_seed_mode != "per_layer":
            seed_states = [s.float() for s in seed_states]

        for i, layer in enumerate(self.layers):
            t = self._sub_type[i]
            if t == "mamba":
                seed_idx = self._sub_seed_idx[i]
                ssm_state0 = seed_states[seed_idx] if seed_idx >= 0 else None
                hidden_states = layer(hidden_states, num_reqs, ssm_state0)
            elif t == "attention":
                hidden_states = layer(positions, hidden_states)
            else:
                hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MambaAttnHybridQwen3ForCausalLM(Qwen3ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
        self.model = MambaAttnHybridQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        latent_seed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds, latent_seed)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.compute_draft_logits(hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    @property
    def has_markov_head(self) -> bool:
        return self.model.markov_head is not None

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    @property
    def sliding_attention_layer_names(self) -> set[str]:
        return self.model.sliding_attention_layer_names

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash should use mask_token_id to embed the padding hidden state"
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = ["mask_embedding", "confidence_head"]
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        skip_substrs.append("verifier_lm_head")
        skip_substrs.append("verifier_norm")

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
