# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


_DFLASH_VALID_LAYER_TYPES = frozenset({"full_attention", "sliding_attention"})


def _get_dflash_layer_types(config: Qwen3Config) -> tuple[str, ...]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return ("full_attention",) * config.num_hidden_layers
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"DFlash layer_types length {len(layer_types)} does not match "
            f"num_hidden_layers {config.num_hidden_layers}."
        )
    invalid = set(layer_types) - _DFLASH_VALID_LAYER_TYPES
    if invalid:
        raise ValueError(f"Invalid DFlash layer_type(s): {sorted(invalid)}.")
    if "sliding_attention" in layer_types and not getattr(
        config, "sliding_window", None
    ):
        raise ValueError(
            "DFlash sliding_attention layers require `sliding_window` in config."
        )
    return tuple(layer_types)


class _RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(
        self, x: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        if gate is not None:
            x = x * F.silu(gate.float())
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


def _ssd_reference(x, dt, A, B, C, D, initial_states):
    b, L, h, p = x.shape
    g, n = B.shape[2], B.shape[3]
    rep = h // g
    if g == 1:
        B = B.expand(b, L, h, n)
        C = C.expand(b, L, h, n)
    else:
        B = B.repeat_interleave(rep, dim=2)
        C = C.repeat_interleave(rep, dim=2)
    state = initial_states
    ys = []
    for t in range(L):
        dA = torch.exp(dt[:, t] * A)
        dBx = dt[:, t][..., None, None] * (x[:, t][..., None] * B[:, t][:, :, None, :])
        state = dA[..., None, None] * state + dBx
        y = (state * C[:, t][:, :, None, :]).sum(-1) + D[None, :, None] * x[:, t]
        ys.append(y)
    return torch.stack(ys, dim=1)


def _dflash_hybrid_ssd_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    num_blocks: int,
    initial_states: torch.Tensor | None = None,
) -> torch.Tensor:
    block_len = x.shape[0] // num_blocks
    chunk_size = next_power_of_2(block_len)
    cu_seqlens = (
        torch.arange(num_blocks + 1, device=x.device, dtype=torch.int32) * block_len
    )
    seq_idx = torch.arange(num_blocks, device=x.device, dtype=torch.int32)
    out = torch.empty_like(x)
    mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_seqlens,
        last_chunk_indices=seq_idx,
        seq_idx=seq_idx,
        out=out,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dt_softplus=True,
        state_dtype=torch.float32,
    )
    return out


def _dflash_hybrid_ssd_scan_fake(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    num_blocks: int,
    initial_states: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="dflash_hybrid_ssd_scan",
    op_func=_dflash_hybrid_ssd_scan,
    fake_impl=_dflash_hybrid_ssd_scan_fake,
)


def _read_ssd_reference_max_l(default: int = 16, cap: int = 64) -> int:
    raw = os.environ.get("HYBRID_SSD_REFERENCE_MAX_L")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "HYBRID_SSD_REFERENCE_MAX_L=%r is not an integer; using default %d.",
            raw,
            default,
        )
        return default
    return max(0, min(value, cap))


_SSD_REFERENCE_MAX_L = _read_ssd_reference_max_l()


class MambaDraftMixer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        d_state: int,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        conv_kernel: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.d_inner = num_heads * head_dim
        self.num_heads, self.head_dim = num_heads, head_dim
        self.n_groups, self.d_state = n_groups, d_state
        self.conv_kernel = conv_kernel
        self.conv_dim = self.d_inner + 2 * n_groups * d_state

        self.in_proj = nn.Linear(
            hidden_size,
            2 * self.d_inner + 2 * n_groups * d_state + num_heads,
            bias=False,
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, conv_kernel,
            groups=self.conv_dim, padding=conv_kernel - 1, bias=True,
        )
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, num_heads + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))
        self.norm = _RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, L, _ = hidden_states.shape
        gate, xBC, dt = torch.split(
            self.in_proj(hidden_states),
            [self.d_inner, self.conv_dim, self.num_heads],
            dim=-1,
        )
        _kc = self.conv_kernel
        _dtype = xBC.dtype
        _w = self.conv1d.weight.squeeze(1).float()
        _xpad = F.pad(xBC.float(), (0, 0, _kc - 1, 0))
        _conv = self.conv1d.bias.float() + sum(
            _w[:, k] * _xpad[:, k : k + L, :] for k in range(_kc)
        )
        xBC = F.silu(_conv.to(_dtype))
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())

        if hidden_states.is_cuda and L > _SSD_REFERENCE_MAX_L:
            T = b * L
            y = torch.ops.vllm.dflash_hybrid_ssd_scan(
                x.reshape(T, self.num_heads, self.head_dim),
                dt.reshape(T, self.num_heads),
                A,
                B.reshape(T, self.n_groups, self.d_state),
                C.reshape(T, self.n_groups, self.d_state),
                self.D,
                self.dt_bias,
                b,
                None if initial_states is None else initial_states.to(torch.float32),
            )
            y = y.reshape(b, L, self.d_inner)
        else:
            if initial_states is None:
                initial_states = torch.zeros(
                    b, self.num_heads, self.head_dim, self.d_state,
                    dtype=torch.float32, device=hidden_states.device,
                )
            dt_r = F.softplus(dt.float() + self.dt_bias.float())
            y = _ssd_reference(
                x.reshape(b, L, self.num_heads, self.head_dim).float(),
                dt_r,
                A,
                B.reshape(b, L, self.n_groups, self.d_state).float(),
                C.reshape(b, L, self.n_groups, self.d_state).float(),
                self.D.float(),
                initial_states.float(),
            )
            y = y.to(hidden_states.dtype).reshape(b, L, self.d_inner)

        y = self.norm(y, gate)
        return self.out_proj(y)


class DFlashAttention(Attention):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if isinstance(spec, SlidingWindowSpec):
            return FullAttentionSpec(
                block_size=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                head_size_v=getattr(spec, "head_size_v", spec.head_size),
                dtype=spec.dtype,
                kv_quant_mode=spec.kv_quant_mode,
                page_size_padded=spec.page_size_padded,
            )
        return spec

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        saved = self.kv_sharing_target_layer_name
        self.kv_sharing_target_layer_name = None
        result = super().forward(query, key, value, output_shape)
        self.kv_sharing_target_layer_name = saved
        return result


class DFlashQwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        sliding_window: int | None = None,
        qk_norm: bool = True,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )
        self.attn = DFlashAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
        )
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.qk_norm:
            q_shape, k_shape = q.shape, k.shape
            q = self.q_norm(
                q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
            ).view(q_shape)
            k = self.k_norm(
                k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
            ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
