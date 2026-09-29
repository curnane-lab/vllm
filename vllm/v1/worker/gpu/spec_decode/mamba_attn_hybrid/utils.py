# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

logger = init_logger(__name__)


def load_mamba_attn_hybrid_model(
    target_model: nn.Module, vllm_config: VllmConfig
) -> nn.Module:
    from vllm.compilation.backends import set_model_tag

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=True,
            backend=speculative_config.attention_backend,
        ),
    )

    with set_model_tag("mamba_attn_hybrid_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    if get_pp_group().world_size != 1:
        raise NotImplementedError(
            "mamba_attn_hybrid does not support pipeline parallelism."
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if target_embed is not None and _should_share(
        draft_model, "has_own_embed_tokens", draft_embed, target_embed
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = getattr(target_model, "lm_head", None)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model


def wire_zero_copy_kv(draft_model: nn.Module, target_model: nn.Module) -> list[int]:
    hf_config = draft_model.config
    attn_kv_layer_ids = list(getattr(hf_config, "attn_kv_layer_ids", None) or [])
    attn_sublayers = draft_model.model.attn_sublayers
    if len(attn_kv_layer_ids) != len(attn_sublayers):
        raise ValueError(
            f"attn_kv_layer_ids has {len(attn_kv_layer_ids)} entries but the recipe "
            f"has {len(attn_sublayers)} attention sub-layer(s); they must match."
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    n_verifier_layers = len(target_language_model.model.layers)
    bad = [i for i in attn_kv_layer_ids if not 0 <= i < n_verifier_layers]
    if bad:
        raise ValueError(
            f"attn_kv_layer_ids {bad} out of range for a verifier with "
            f"{n_verifier_layers} layers (valid 0..{n_verifier_layers - 1})."
        )

    for sub, verifier_lid in zip(attn_sublayers, attn_kv_layer_ids):
        sub.self_attn.attn.kv_sharing_target_layer_name = (
            f"model.layers.{verifier_lid}.self_attn.attn"
        )
    return attn_kv_layer_ids
