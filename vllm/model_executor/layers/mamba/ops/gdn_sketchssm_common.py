# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM storage, q/k rotation and cold sketch build for Gated DeltaNet.

The state ``(slot, HV, V, K)`` is kept in rotated key coordinates. Per
request, a value head of rank ``m`` (0 = dense) keeps packed BF16 rows ``u``
(leading key columns), ``phi`` (coefficient map) and ``fs`` (projected erase
history). Sketch buffers are indexed by the persistent request index.
"""

from dataclasses import dataclass

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

GDN_SKETCH_PIVOTS = 4
# Default window; the kernels take any multiple of GDN_SKETCH_WINDOW_ALIGN.
GDN_SKETCH_WINDOW = 16
GDN_SKETCH_WINDOW_ALIGN = 16
GDN_SKETCH_HEAD_DIM = 128


def gdn_sketch_window_supported(window: int) -> bool:
    """Whether the GDN SketchSSM kernels take this window length."""
    return window >= GDN_SKETCH_WINDOW_ALIGN and window % GDN_SKETCH_WINDOW_ALIGN == 0


def gdn_sketch_layout(
    ranks: torch.Tensor, window: int = GDN_SKETCH_WINDOW
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Per-head ``(u_off, phi_off, fs_off, FG)`` table and packed row lengths."""
    if not gdn_sketch_window_supported(window):
        raise ValueError(
            f"GDN SketchSSM window {window} is not a multiple of "
            f"{GDN_SKETCH_WINDOW_ALIGN}"
        )
    k, p = GDN_SKETCH_HEAD_DIM, GDN_SKETCH_PIVOTS
    rows = []
    nu = nm = nf = 0
    for m in ranks.tolist():
        if not 0 <= m <= k:
            raise ValueError(f"GDN sketch rank {m} outside [0, {k}]")
        # AscendC vector ops require 32-byte-aligned row starts; pad the
        # feature grain to 16 bf16 elements so every packed row (fs at fg
        # stride, phi gains at (p+1)*fg) is 32B-aligned in UB.
        fg = max(16, (m + 15) // 16 * 16)
        rows.append((nu, nm, nf, fg))
        nu += m * k
        nm += 0 if m in (0, k) else m * k if m <= p else p * k + (p + 1) * fg
        nf += window * fg if m else 0
    # Rows are staged with 32-byte copies.
    sizes = tuple(max(16, (n + 15) // 16 * 16) for n in (nu, nm, nf))
    return torch.tensor(rows, dtype=torch.int32), sizes


def gdn_rank_cap(ranks: torch.Tensor) -> int:
    """Rank bound ``G`` the kernels are specialized on (multiple of 8)."""
    return max(8, (int(ranks.max()) + 7) // 8 * 8)


class GDNSketchTables(torch.nn.Module):
    """Ranks and packed-layout table of one GDN layer."""

    def __init__(
        self, ranks: torch.Tensor, num_k_heads: int, window: int = GDN_SKETCH_WINDOW
    ):
        super().__init__()
        ranks = ranks.to(torch.int32).cpu()
        layout, self.sizes = gdn_sketch_layout(ranks, window)
        self.num_k_heads = num_k_heads
        self.window = window
        self.num_v_heads = ranks.numel()
        self.rank_cap = gdn_rank_cap(ranks)
        device = torch.get_default_device()
        self.register_buffer("ranks", ranks.to(device), persistent=False)
        self.register_buffer("layout", layout.to(device), persistent=False)

    def buffer_specs(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Per-request ``name: (shape, dtype)`` of the sketch buffers."""
        nu, nm, nf = self.sizes
        hv, h = self.num_v_heads, self.num_k_heads
        d, w = GDN_SKETCH_HEAD_DIM, self.window
        return {
            "u": ((nu,), torch.bfloat16),
            "phi": ((nm,), torch.bfloat16),
            "fs": ((nf,), torch.bfloat16),
            "beta": ((hv, w), torch.float32),
            "current_d": ((hv, d), torch.float32),
            "current_k": ((h, d), torch.float32),
        }


@dataclass
class GDNSketchArgs:
    """One GDN layer's per-request sketch buffers (first dim: request index)."""

    u: torch.Tensor
    phi: torch.Tensor
    fs: torch.Tensor
    beta: torch.Tensor
    current_d: torch.Tensor
    current_k: torch.Tensor
    tables: GDNSketchTables

    @classmethod
    def allocate(
        cls, tables: GDNSketchTables, max_num_reqs: int, device=None
    ) -> "GDNSketchArgs":
        buffers = {
            name: torch.zeros(max_num_reqs, *shape, dtype=dtype, device=device)
            for name, (shape, dtype) in tables.buffer_specs().items()
        }
        return cls(**buffers, tables=tables)


@triton.jit(do_not_specialize=["T"])
def _rotate_qk_kernel(X, RT, T, s_t, HK: tl.constexpr, K: tl.constexpr,
                      BT: tl.constexpr,
                      DOT_PRECISION: tl.constexpr):  # fmt: skip
    # In place, q then k of key head h: x <- R x.
    pt = tl.program_id(0)
    h = tl.program_id(1)
    rows = pt * BT + tl.arange(0, BT)
    cols = tl.arange(0, K)
    rt = tl.load(RT + h * K * K + cols[:, None] * K + cols[None, :])
    for z in tl.static_range(2):
        ptr = X + rows[:, None].to(tl.int64) * s_t + (z * HK + h) * K + cols[None, :]
        x = tl.load(ptr, mask=rows[:, None] < T, other=0.0).to(tl.float32)
        y = tl.dot(x, rt, input_precision=DOT_PRECISION)
        tl.store(ptr, y.to(X.dtype.element_ty), mask=rows[:, None] < T)


def gdn_rotation_from_frames(frames: torch.Tensor) -> torch.Tensor:
    """``(H, K, K)`` FP32 ``R^T`` of a layer's per-key-head frames."""
    return frames.to(torch.float32).transpose(-1, -2).contiguous()


def gdn_sketch_rotate_(mixed_qkv: torch.Tensor, frames_t: torch.Tensor) -> None:
    """Rotate q and k of ``mixed_qkv (tokens, 2 H K + HV V)`` in place.

    ``frames_t`` is ``gdn_rotation_from_frames(frames)``.
    """
    num_tokens = mixed_qkv.shape[0]
    if num_tokens == 0:
        return
    h, k, _ = frames_t.shape
    assert mixed_qkv.stride(1) == 1 and frames_t.is_contiguous()
    assert frames_t.dtype == torch.float32
    bt = 64 if num_tokens >= 1024 else 32
    _rotate_qk_kernel[(triton.cdiv(num_tokens, bt), h)](
        mixed_qkv, frames_t, num_tokens, mixed_qkv.stride(0), HK=h, K=k, BT=bt,
        DOT_PRECISION="ieee" if current_platform.device_type == "npu" else "tf32x3",
        num_warps=4,
    )  # fmt: skip


@triton.jit
def _build_head(base, sk, h, u, packed, fs, beta, widths, layout, SU: tl.constexpr,
                SM: tl.constexpr, SF: tl.constexpr, SV: tl.constexpr,
                SK: tl.constexpr, HV: tl.constexpr, G: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, W: tl.constexpr):  # fmt: skip
    # Builds one head's sketch and resets its erase history and gates.
    j = tl.arange(0, K)
    v = tl.arange(0, V)
    w = tl.arange(0, 16)
    sk = sk.to(tl.int64)
    for wr in tl.static_range(0, W, 16):
        tl.store(beta + (sk * HV + h) * W + wr + w, tl.zeros([16], tl.float32))
    m = tl.load(widths + h)
    if m > 0:
        uoff = tl.load(layout + h * 4)
        moff = tl.load(layout + h * 4 + 1)
        foff = tl.load(layout + h * 4 + 2)
        fg = tl.load(layout + h * 4 + 3)
        o = tl.arange(0, 16 * K)
        for wr in tl.static_range(0, W, 16):
            tl.store(
                fs + sk * SF + foff + wr * fg + o,
                tl.zeros([16 * K], fs.dtype.element_ty),
                o < 16 * fg,
            )
        s = tl.load(base + v[:, None] * SV + j[None, :] * SK).to(tl.float32)
        tl.store(
            u + sk * SU + uoff + j[None, :] * V + v[:, None],
            s,
            (j[None, :] < m) & (j[None, :] < G),
        )
        if m < K:
            energy = tl.sum(s * s, axis=0)
            mean = tl.sum(energy) / K
            safe_mean = tl.where(mean > 0.0, mean, 1.0)
            u0 = tl.load(base + v * SV + 0 * SK).to(tl.float32)
            v0 = tl.full((V,), 0.0, tl.float32)
            if m > 0:
                w0 = u0
                e0 = tl.sum(w0 * w0)
                keep0 = e0 > 0.0
                v0 = tl.where(keep0, w0 / tl.sqrt(tl.where(keep0, e0, 1.0)), 0.0)
            z0 = tl.sum(v0[:, None] * s, axis=0) / tl.sqrt(safe_mean)
            u1 = tl.load(base + v * SV + 1 * SK).to(tl.float32)
            v1 = tl.full((V,), 0.0, tl.float32)
            if m > 1:
                w1 = u1
                w1 = w1 - v0 * tl.sum(v0 * w1)
                w1 = w1 - v0 * tl.sum(v0 * w1)
                e1 = tl.sum(w1 * w1)
                keep1 = e1 > 1.0e-12 * tl.sum(u1 * u1)
                v1 = tl.where(keep1, w1 / tl.sqrt(tl.where(keep1, e1, 1.0)), 0.0)
            z1 = tl.sum(v1[:, None] * s, axis=0) / tl.sqrt(safe_mean)
            u2 = tl.load(base + v * SV + 2 * SK).to(tl.float32)
            v2 = tl.full((V,), 0.0, tl.float32)
            if m > 2:
                w2 = u2
                w2 = w2 - v0 * tl.sum(v0 * w2)
                w2 = w2 - v1 * tl.sum(v1 * w2)
                w2 = w2 - v0 * tl.sum(v0 * w2)
                w2 = w2 - v1 * tl.sum(v1 * w2)
                e2 = tl.sum(w2 * w2)
                keep2 = e2 > 1.0e-12 * tl.sum(u2 * u2)
                v2 = tl.where(keep2, w2 / tl.sqrt(tl.where(keep2, e2, 1.0)), 0.0)
            z2 = tl.sum(v2[:, None] * s, axis=0) / tl.sqrt(safe_mean)
            u3 = tl.load(base + v * SV + 3 * SK).to(tl.float32)
            v3 = tl.full((V,), 0.0, tl.float32)
            if m > 3:
                w3 = u3
                w3 = w3 - v0 * tl.sum(v0 * w3)
                w3 = w3 - v1 * tl.sum(v1 * w3)
                w3 = w3 - v2 * tl.sum(v2 * w3)
                w3 = w3 - v0 * tl.sum(v0 * w3)
                w3 = w3 - v1 * tl.sum(v1 * w3)
                w3 = w3 - v2 * tl.sum(v2 * w3)
                e3 = tl.sum(w3 * w3)
                keep3 = e3 > 1.0e-12 * tl.sum(u3 * u3)
                v3 = tl.where(keep3, w3 / tl.sqrt(tl.where(keep3, e3, 1.0)), 0.0)
            z3 = tl.sum(v3[:, None] * s, axis=0) / tl.sqrt(safe_mean)
            residual = tl.maximum(
                energy / safe_mean - z0 * z0 - z1 * z1 - z2 * z2 - z3 * z3, 0.0
            )
            residual = tl.where(j < tl.minimum(m, 4), 0.0, residual)
            denominator = residual + 0.1
            a = residual / denominator
            b0 = tl.where(j < m, z0 / denominator, 0.0)
            b1 = tl.where(j < m, z1 / denominator, 0.0)
            b2 = tl.where(j < m, z2 / denominator, 0.0)
            b3 = tl.where(j < m, z3 / denominator, 0.0)
            l0_0 = tl.sqrt(1.0 + tl.sum(z0 * b0))
            l1_0 = (tl.sum(z1 * b0)) / l0_0
            l1_1 = tl.sqrt(1.0 + tl.sum(z1 * b1) - l1_0 * l1_0)
            l2_0 = (tl.sum(z2 * b0)) / l0_0
            l2_1 = (tl.sum(z2 * b1) - l2_0 * l1_0) / l1_1
            l2_2 = tl.sqrt(1.0 + tl.sum(z2 * b2) - l2_0 * l2_0 - l2_1 * l2_1)
            l3_0 = (tl.sum(z3 * b0)) / l0_0
            l3_1 = (tl.sum(z3 * b1) - l3_0 * l1_0) / l1_1
            l3_2 = (tl.sum(z3 * b2) - l3_0 * l2_0 - l3_1 * l2_1) / l2_2
            l3_3 = tl.sqrt(
                1.0 + tl.sum(z3 * b3) - l3_0 * l3_0 - l3_1 * l3_1 - l3_2 * l3_2
            )
            y0 = (b0) / l0_0
            y1 = (b1 - l1_0 * y0) / l1_1
            y2 = (b2 - l2_0 * y0 - l2_1 * y1) / l2_2
            y3 = (b3 - l3_0 * y0 - l3_1 * y1 - l3_2 * y2) / l3_3
            g3 = (y3) / l3_3
            g2 = (y2 - l3_2 * g3) / l2_2
            g1 = (y1 - l2_1 * g2 - l3_1 * g3) / l1_1
            g0 = (y0 - l1_0 * g1 - l2_0 * g2 - l3_0 * g3) / l0_0
            factor = tl.where(j < m, 0.1 / denominator, 1.0)
            tbase = packed + sk * SM + moff + j
            index = sk * SM + moff + 4 * K + j
            if m <= 4:
                for n in range(m):
                    f0 = tl.sum(tl.where(j == n, g0, 0.0))
                    f1 = tl.sum(tl.where(j == n, g1, 0.0))
                    f2 = tl.sum(tl.where(j == n, g2, 0.0))
                    f3 = tl.sum(tl.where(j == n, g3, 0.0))
                    merged = (f0 * z0 + f1 * z1 + f2 * z2 + f3 * z3) * factor
                    tl.store(tbase + n * K, merged)
            else:
                tl.store(packed + index, a, j < m)
                tl.store(tbase + 0 * K, z0 * factor)
                tl.store(packed + index + 1 * fg, g0, j < m)
                tl.store(tbase + 1 * K, z1 * factor)
                tl.store(packed + index + 2 * fg, g1, j < m)
                tl.store(tbase + 2 * K, z2 * factor)
                tl.store(packed + index + 3 * fg, g2, j < m)
                tl.store(tbase + 3 * K, z3 * factor)
                tl.store(packed + index + 4 * fg, g3, j < m)


@triton.jit
def _gdn_build_kernel(state, u, packed, fs, beta, widths, layout, slots, metas,
                      flags, rows, batch, null_block_id, SB: tl.constexpr,
                      SH: tl.constexpr, SV: tl.constexpr, SK: tl.constexpr,
                      SU: tl.constexpr, SM: tl.constexpr, SF: tl.constexpr,
                      HV: tl.constexpr, G: tl.constexpr, K: tl.constexpr,
                      V: tl.constexpr, W: tl.constexpr,
                      ROW_LIST: tl.constexpr):  # fmt: skip
    # With ROW_LIST, programs walk ``rows`` and stop at the first -1 padding.
    h = tl.program_id(1)
    it = tl.program_id(0)
    if ROW_LIST:
        row = tl.load(rows + it, mask=it < batch, other=-1)
    else:
        row = tl.where(tl.load(flags + it) != 0, it, -1)
    while row >= 0:
        slot = tl.load(slots + row).to(tl.int32)
        if (slot != null_block_id) & (slot >= 0):
            sk = tl.load(metas + row)
            _build_head(state + slot.to(tl.int64) * SB + h.to(tl.int64) * SH, sk, h,
                        u, packed, fs, beta, widths, layout, SU, SM, SF, SV, SK,
                        HV, G, K, V, W)  # fmt: skip
        if ROW_LIST:
            it += tl.num_programs(0)
            row = tl.load(rows + it, mask=it < batch, other=-1)
        else:
            row = row * 0 - 1


@triton.jit
def _gdn_build_kernel_v2(state, u, packed, fs, beta, widths, layout, slots,
                         metas, flags, rows, batch, null_block_id, SB, SH, SV,
                         SK, SU, SM, SF, HV: tl.constexpr, G: tl.constexpr,
                         K: tl.constexpr, V: tl.constexpr, W: tl.constexpr,
                         BV: tl.constexpr, FS2: tl.constexpr,
                         ROW_LIST: tl.constexpr):  # fmt: skip
    # Chunked-over-V rebuild of _gdn_build_head: the state tile is streamed in
    # (BV, K) blocks so the live UB set stays a few dozen KB instead of the
    # 2 MB that defeated triton-ascend's planner on the unchunked kernel.
    h = tl.program_id(1)
    it = tl.program_id(0)
    if ROW_LIST:
        row = tl.load(rows + it, mask=it < batch, other=-1)
    else:
        row = tl.where(tl.load(flags + it) != 0, it, -1)
    if row >= 0:
        slot = tl.load(slots + row).to(tl.int32)
        if (slot != null_block_id) & (slot >= 0):
            sk = tl.load(metas + row).to(tl.int64)
            w = tl.arange(0, W)
            tl.store(beta + (sk * HV + h) * W + w, tl.zeros([W], tl.float32))
            m = tl.load(widths + h)
            if m > 0:
                uoff = tl.load(layout + h * 4)
                moff = tl.load(layout + h * 4 + 1)
                foff = tl.load(layout + h * 4 + 2)
                fg = tl.load(layout + h * 4 + 3)
                fo = tl.arange(0, FS2)
                tl.store(
                    fs + sk * SF + foff + fo,
                    tl.zeros([FS2], fs.dtype.element_ty),
                    fo < W * fg,
                )
                base = state + slot.to(tl.int64) * SB + h.to(tl.int64) * SH
                vv = tl.arange(0, V)
                kk = tl.arange(0, K)
                kk4 = tl.arange(0, 4)
                cols = tl.load(base + vv[:, None] * SV + kk4[None, :] * SK).to(
                    tl.float32
                )
                # Four Gram-Schmidt pivots over V, mirroring the torch
                # fallback's double reorthogonalization and keep thresholds.
                col0 = tl.sum(tl.where(kk4[None, :] == 0, cols, 0.0), axis=1)
                col1 = tl.sum(tl.where(kk4[None, :] == 1, cols, 0.0), axis=1)
                col2 = tl.sum(tl.where(kk4[None, :] == 2, cols, 0.0), axis=1)
                col3 = tl.sum(tl.where(kk4[None, :] == 3, cols, 0.0), axis=1)
                w0 = col0
                e0 = tl.sum(w0 * w0)
                keep0 = e0 > 0.0
                v0 = tl.where(keep0, w0 / tl.sqrt(tl.where(keep0, e0, 1.0)), 0.0)
                w1 = col1
                w1 = w1 - v0 * tl.sum(v0 * w1)
                w1 = w1 - v0 * tl.sum(v0 * w1)
                e1 = tl.sum(w1 * w1)
                keep1 = e1 > 1.0e-12 * tl.sum(col1 * col1)
                ok1 = keep1 & (m > 1)
                v1 = tl.where(ok1, w1 / tl.sqrt(tl.where(ok1, e1, 1.0)), 0.0)
                w2 = col2
                w2 = w2 - v0 * tl.sum(v0 * w2)
                w2 = w2 - v1 * tl.sum(v1 * w2)
                w2 = w2 - v0 * tl.sum(v0 * w2)
                w2 = w2 - v1 * tl.sum(v1 * w2)
                e2 = tl.sum(w2 * w2)
                keep2 = e2 > 1.0e-12 * tl.sum(col2 * col2)
                ok2 = keep2 & (m > 2)
                v2 = tl.where(ok2, w2 / tl.sqrt(tl.where(ok2, e2, 1.0)), 0.0)
                w3 = col3
                w3 = w3 - v0 * tl.sum(v0 * w3)
                w3 = w3 - v1 * tl.sum(v1 * w3)
                w3 = w3 - v2 * tl.sum(v2 * w3)
                w3 = w3 - v0 * tl.sum(v0 * w3)
                w3 = w3 - v1 * tl.sum(v1 * w3)
                w3 = w3 - v2 * tl.sum(v2 * w3)
                e3 = tl.sum(w3 * w3)
                keep3 = e3 > 1.0e-12 * tl.sum(col3 * col3)
                ok3 = keep3 & (m > 3)
                v3 = tl.where(ok3, w3 / tl.sqrt(tl.where(ok3, e3, 1.0)), 0.0)
                z0 = tl.zeros([K], tl.float32)
                z1 = tl.zeros([K], tl.float32)
                z2 = tl.zeros([K], tl.float32)
                z3 = tl.zeros([K], tl.float32)
                energy = tl.zeros([K], tl.float32)
                um = kk[None, :] < tl.minimum(m, G)
                for v0c in tl.static_range(0, V, BV):
                    vc = v0c + tl.arange(0, BV)
                    s_chunk = tl.load(
                        base + vc[:, None] * SV + kk[None, :] * SK
                    ).to(tl.float32)
                    tl.store(
                        u + sk * SU + uoff + kk[None, :] * V + vc[:, None],
                        s_chunk,
                        um,
                    )
                    energy += tl.sum(s_chunk * s_chunk, axis=0)
                    sel = vc[None, :] == vv[:, None]
                    v0c_v = tl.sum(tl.where(sel, v0[:, None], 0.0), axis=0)
                    v1c_v = tl.sum(tl.where(sel, v1[:, None], 0.0), axis=0)
                    v2c_v = tl.sum(tl.where(sel, v2[:, None], 0.0), axis=0)
                    v3c_v = tl.sum(tl.where(sel, v3[:, None], 0.0), axis=0)
                    z0 += tl.sum(v0c_v[:, None] * s_chunk, axis=0)
                    z1 += tl.sum(v1c_v[:, None] * s_chunk, axis=0)
                    z2 += tl.sum(v2c_v[:, None] * s_chunk, axis=0)
                    z3 += tl.sum(v3c_v[:, None] * s_chunk, axis=0)
                mean = tl.sum(energy) / K
                safe_mean = tl.where(mean > 0.0, mean, 1.0)
                smrt = tl.sqrt(safe_mean)
                z0 = z0 / smrt
                z1 = z1 / smrt
                z2 = z2 / smrt
                z3 = z3 / smrt
                residual = tl.maximum(
                    energy / safe_mean - z0 * z0 - z1 * z1 - z2 * z2 - z3 * z3,
                    0.0,
                )
                residual = tl.where(kk < tl.minimum(m, 4), 0.0, residual)
                denominator = residual + 0.1
                a = residual / denominator
                jm = kk < m
                b0 = tl.where(jm, z0 / denominator, 0.0)
                b1 = tl.where(jm, z1 / denominator, 0.0)
                b2 = tl.where(jm, z2 / denominator, 0.0)
                b3 = tl.where(jm, z3 / denominator, 0.0)
                l0_0 = tl.sqrt(1.0 + tl.sum(z0 * b0))
                l1_0 = tl.sum(z1 * b0) / l0_0
                l1_1 = tl.sqrt(1.0 + tl.sum(z1 * b1) - l1_0 * l1_0)
                l2_0 = tl.sum(z2 * b0) / l0_0
                l2_1 = (tl.sum(z2 * b1) - l2_0 * l1_0) / l1_1
                l2_2 = tl.sqrt(1.0 + tl.sum(z2 * b2) - l2_0 * l2_0 - l2_1 * l2_1)
                l3_0 = tl.sum(z3 * b0) / l0_0
                l3_1 = (tl.sum(z3 * b1) - l3_0 * l1_0) / l1_1
                l3_2 = (tl.sum(z3 * b2) - l3_0 * l2_0 - l3_1 * l2_1) / l2_2
                l3_3 = tl.sqrt(
                    1.0 + tl.sum(z3 * b3) - l3_0 * l3_0 - l3_1 * l3_1 - l3_2 * l3_2
                )
                y0 = b0 / l0_0
                y1 = (b1 - l1_0 * y0) / l1_1
                y2 = (b2 - l2_0 * y0 - l2_1 * y1) / l2_2
                y3 = (b3 - l3_0 * y0 - l3_1 * y1 - l3_2 * y2) / l3_3
                g3 = y3 / l3_3
                g2 = (y2 - l3_2 * g3) / l2_2
                g1 = (y1 - l2_1 * g2 - l3_1 * g3) / l1_1
                g0 = (y0 - l1_0 * g1 - l2_0 * g2 - l3_0 * g3) / l0_0
                factor = tl.where(jm, 0.1 / denominator, 1.0)
                tbase = packed + sk * SM + moff
                if m <= 4:
                    for n in tl.static_range(4):
                        if m > n:
                            f0 = tl.sum(tl.where(kk == n, g0, 0.0))
                            f1 = tl.sum(tl.where(kk == n, g1, 0.0))
                            f2 = tl.sum(tl.where(kk == n, g2, 0.0))
                            f3 = tl.sum(tl.where(kk == n, g3, 0.0))
                            merged = (
                                f0 * z0 + f1 * z1 + f2 * z2 + f3 * z3
                            ) * factor
                            tl.store(tbase + n * K + kk, merged)
                else:
                    zm = kk < m
                    tl.store(tbase + 4 * K + kk, tl.where(zm, a, 0.0), kk < fg)
                    tl.store(tbase + 0 * K + kk, z0 * factor)
                    tl.store(tbase + 4 * K + 1 * fg + kk, tl.where(zm, g0, 0.0), kk < fg)
                    tl.store(tbase + 1 * K + kk, z1 * factor)
                    tl.store(tbase + 4 * K + 2 * fg + kk, tl.where(zm, g1, 0.0), kk < fg)
                    tl.store(tbase + 2 * K + kk, z2 * factor)
                    tl.store(tbase + 4 * K + 3 * fg + kk, tl.where(zm, g2, 0.0), kk < fg)
                    tl.store(tbase + 3 * K + kk, z3 * factor)
                    tl.store(tbase + 4 * K + 4 * fg + kk, tl.where(zm, g3, 0.0), kk < fg)


def gdn_sketch_build_v2(
    state: torch.Tensor,
    flags: torch.Tensor,
    slots: torch.Tensor,
    meta: torch.Tensor,
    sketch: GDNSketchArgs,
    null_block_id: int = NULL_BLOCK_ID,
    rows: torch.Tensor | None = None,
) -> None:
    """Chunked triton build for NPU: one program per (row, head)."""
    batch = flags.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    _, hv, v, k = state.shape
    programs = batch
    _gdn_build_kernel_v2[(programs, hv)](
        state, sketch.u, sketch.phi, sketch.fs, sketch.beta, t.ranks, t.layout,
        slots, meta, flags, flags if rows is None else rows, batch,
        null_block_id, SB=state.stride(0), SH=state.stride(1),
        SV=state.stride(2), SK=state.stride(3), SU=sketch.u.stride(0),
        SM=sketch.phi.stride(0), SF=sketch.fs.stride(0), HV=hv, G=t.rank_cap,
        K=k, V=v, W=t.window, BV=32,
        FS2=triton.next_power_of_2(t.window * t.rank_cap),
        ROW_LIST=rows is not None, num_warps=2,
    )  # fmt: skip


@triton.jit
def _gdn_build_kernel_v3(state, u, packed, fs, beta, widths, layout, slots,
                         metas, flags, rows, batch, null_block_id, SB, SH, SV,
                         SK, SU, SM, SF, HV: tl.constexpr, G: tl.constexpr,
                         K: tl.constexpr, V: tl.constexpr, W: tl.constexpr,
                         BV: tl.constexpr, FS2: tl.constexpr,
                         ROW_LIST: tl.constexpr):  # fmt: skip
    # Sel-free rebuild. The four pivot projections of the state are
    # accumulated as raw column products g_j = c_j^T S (linear in S), and the
    # Gram-Schmidt projections z_i = v_i^T S are rebuilt from the g_j with
    # scalar coefficients, so no one-hot gather over the chunk rows exists.
    h = tl.program_id(1)
    it = tl.program_id(0)
    if ROW_LIST:
        row = tl.load(rows + it, mask=it < batch, other=-1)
    else:
        row = tl.where(tl.load(flags + it) != 0, it, -1)
    while row >= 0:
        slot = tl.load(slots + row).to(tl.int32)
        if (slot != null_block_id) & (slot >= 0):
            sk = tl.load(metas + row).to(tl.int64)
            w = tl.arange(0, W)
            tl.store(beta + (sk * HV + h) * W + w, tl.zeros([W], tl.float32))
            m = tl.load(widths + h)
            if m > 0:
                uoff = tl.load(layout + h * 4)
                moff = tl.load(layout + h * 4 + 1)
                foff = tl.load(layout + h * 4 + 2)
                fg = tl.load(layout + h * 4 + 3)
                fo = tl.arange(0, FS2)
                tl.store(
                    fs + sk * SF + foff + fo,
                    tl.zeros([FS2], fs.dtype.element_ty),
                    fo < W * fg,
                )
                base = state + slot.to(tl.int64) * SB + h.to(tl.int64) * SH
                vv = tl.arange(0, V)
                kk = tl.arange(0, K)
                c0 = tl.load(base + vv * SV).to(tl.float32)
                c1 = tl.load(base + vv * SV + SK).to(tl.float32)
                c2 = tl.load(base + vv * SV + 2 * SK).to(tl.float32)
                c3 = tl.load(base + vv * SV + 3 * SK).to(tl.float32)
                # Gram-Schmidt pivots over V, mirroring the torch fallback's
                # double reorthogonalization and keep thresholds.
                w0 = c0
                e0 = tl.sum(w0 * w0)
                keep0 = e0 > 0.0
                w1 = c1
                w1 = w1 - w0 * (tl.sum(w0 * w1) / tl.where(keep0, e0, 1.0))
                w1 = w1 - w0 * (tl.sum(w0 * w1) / tl.where(keep0, e0, 1.0))
                e1 = tl.sum(w1 * w1)
                keep1 = e1 > 1.0e-12 * tl.sum(c1 * c1)
                ok1 = keep1 & (m > 1)
                w2 = c2
                w2 = w2 - w0 * (tl.sum(w0 * w2) / tl.where(keep0, e0, 1.0))
                w2 = w2 - w1 * (tl.sum(w1 * w2) / tl.where(ok1, e1, 1.0))
                w2 = w2 - w0 * (tl.sum(w0 * w2) / tl.where(keep0, e0, 1.0))
                w2 = w2 - w1 * (tl.sum(w1 * w2) / tl.where(ok1, e1, 1.0))
                e2 = tl.sum(w2 * w2)
                keep2 = e2 > 1.0e-12 * tl.sum(c2 * c2)
                ok2 = keep2 & (m > 2)
                w3 = c3
                w3 = w3 - w0 * (tl.sum(w0 * w3) / tl.where(keep0, e0, 1.0))
                w3 = w3 - w1 * (tl.sum(w1 * w3) / tl.where(ok1, e1, 1.0))
                w3 = w3 - w2 * (tl.sum(w2 * w3) / tl.where(ok2, e2, 1.0))
                w3 = w3 - w0 * (tl.sum(w0 * w3) / tl.where(keep0, e0, 1.0))
                w3 = w3 - w1 * (tl.sum(w1 * w3) / tl.where(ok1, e1, 1.0))
                w3 = w3 - w2 * (tl.sum(w2 * w3) / tl.where(ok2, e2, 1.0))
                e3 = tl.sum(w3 * w3)
                keep3 = e3 > 1.0e-12 * tl.sum(c3 * c3)
                ok3 = keep3 & (m > 3)
                # Scalar GS coefficients (v_i = sum_j a_ij c_j, i <= j).
                iv0 = tl.where(keep0, 1.0 / tl.sqrt(e0), 0.0)
                a01 = tl.sum(c1 * w0) * iv0
                a02 = tl.sum(c2 * w0) * iv0
                a03 = tl.sum(c3 * w0) * iv0
                iv1 = tl.where(ok1, 1.0 / tl.sqrt(e1), 0.0)
                a12 = tl.sum(c2 * w1) * iv1
                a13 = tl.sum(c3 * w1) * iv1
                iv2 = tl.where(ok2, 1.0 / tl.sqrt(e2), 0.0)
                a23 = tl.sum(c3 * w2) * iv2
                g0 = tl.zeros([K], tl.float32)
                g1 = tl.zeros([K], tl.float32)
                g2 = tl.zeros([K], tl.float32)
                g3 = tl.zeros([K], tl.float32)
                energy = tl.zeros([K], tl.float32)
                um = kk[None, :] < tl.minimum(m, G)
                for v0c in tl.static_range(0, V, BV):
                    vc = v0c + tl.arange(0, BV)
                    s_chunk = tl.load(
                        base + vc[:, None] * SV + kk[None, :] * SK
                    ).to(tl.float32)
                    tl.store(
                        u + sk * SU + uoff + kk[None, :] * V + vc[:, None],
                        s_chunk,
                        um,
                    )
                    energy += tl.sum(s_chunk * s_chunk, axis=0)
                    cc0 = tl.load(base + vc * SV).to(tl.float32)
                    cc1 = tl.load(base + vc * SV + SK).to(tl.float32)
                    cc2 = tl.load(base + vc * SV + 2 * SK).to(tl.float32)
                    cc3 = tl.load(base + vc * SV + 3 * SK).to(tl.float32)
                    g0 += tl.sum(s_chunk * cc0[:, None], axis=0)
                    g1 += tl.sum(s_chunk * cc1[:, None], axis=0)
                    g2 += tl.sum(s_chunk * cc2[:, None], axis=0)
                    g3 += tl.sum(s_chunk * cc3[:, None], axis=0)
                z0 = g0 * iv0
                z1 = (g1 - a01 * z0) * iv1
                z2 = (g2 - a02 * z0 - a12 * z1) * iv2
                iv3 = tl.where(ok3, 1.0 / tl.sqrt(e3), 0.0)
                z3 = (g3 - a03 * z0 - a13 * z1 - a23 * z2) * iv3
                mean = tl.sum(energy) / K
                safe_mean = tl.where(mean > 0.0, mean, 1.0)
                smrt = tl.sqrt(safe_mean)
                z0 = z0 / smrt
                z1 = z1 / smrt
                z2 = z2 / smrt
                z3 = z3 / smrt
                residual = tl.maximum(
                    energy / safe_mean - z0 * z0 - z1 * z1 - z2 * z2 - z3 * z3,
                    0.0,
                )
                residual = tl.where(kk < tl.minimum(m, 4), 0.0, residual)
                denominator = residual + 0.1
                a = residual / denominator
                jm = kk < m
                b0 = tl.where(jm, z0 / denominator, 0.0)
                b1 = tl.where(jm, z1 / denominator, 0.0)
                b2 = tl.where(jm, z2 / denominator, 0.0)
                b3 = tl.where(jm, z3 / denominator, 0.0)
                l0_0 = tl.sqrt(1.0 + tl.sum(z0 * b0))
                l1_0 = tl.sum(z1 * b0) / l0_0
                l1_1 = tl.sqrt(1.0 + tl.sum(z1 * b1) - l1_0 * l1_0)
                l2_0 = tl.sum(z2 * b0) / l0_0
                l2_1 = (tl.sum(z2 * b1) - l2_0 * l1_0) / l1_1
                l2_2 = tl.sqrt(1.0 + tl.sum(z2 * b2) - l2_0 * l2_0 - l2_1 * l2_1)
                l3_0 = tl.sum(z3 * b0) / l0_0
                l3_1 = (tl.sum(z3 * b1) - l3_0 * l1_0) / l1_1
                l3_2 = (tl.sum(z3 * b2) - l3_0 * l2_0 - l3_1 * l2_1) / l2_2
                l3_3 = tl.sqrt(
                    1.0 + tl.sum(z3 * b3) - l3_0 * l3_0 - l3_1 * l3_1 - l3_2 * l3_2
                )
                y0 = b0 / l0_0
                y1 = (b1 - l1_0 * y0) / l1_1
                y2 = (b2 - l2_0 * y0 - l2_1 * y1) / l2_2
                y3 = (b3 - l3_0 * y0 - l3_1 * y1 - l3_2 * y2) / l3_3
                h3 = y3 / l3_3
                h2 = (y2 - l3_2 * h3) / l2_2
                h1 = (y1 - l2_1 * h2 - l3_1 * h3) / l1_1
                h0 = (y0 - l1_0 * h1 - l2_0 * h2 - l3_0 * h3) / l0_0
                factor = tl.where(jm, 0.1 / denominator, 1.0)
                tbase = packed + sk * SM + moff
                if m <= 4:
                    for n in tl.static_range(4):
                        if m > n:
                            f0 = tl.sum(tl.where(kk == n, h0, 0.0))
                            f1 = tl.sum(tl.where(kk == n, h1, 0.0))
                            f2 = tl.sum(tl.where(kk == n, h2, 0.0))
                            f3 = tl.sum(tl.where(kk == n, h3, 0.0))
                            merged = (
                                f0 * z0 + f1 * z1 + f2 * z2 + f3 * z3
                            ) * factor
                            tl.store(tbase + n * K + kk, merged)
                else:
                    zm = kk < m
                    tl.store(tbase + 4 * K + kk, tl.where(zm, a, 0.0), kk < fg)
                    tl.store(tbase + 0 * K + kk, z0 * factor)
                    tl.store(tbase + 4 * K + 1 * fg + kk, tl.where(zm, h0, 0.0), kk < fg)
                    tl.store(tbase + 1 * K + kk, z1 * factor)
                    tl.store(tbase + 2 * K + kk, z2 * factor)
                    tl.store(tbase + 4 * K + 2 * fg + kk, tl.where(zm, h1, 0.0), kk < fg)
                    tl.store(tbase + 3 * K + kk, z3 * factor)
                    tl.store(tbase + 4 * K + 3 * fg + kk, tl.where(zm, h2, 0.0), kk < fg)
                    tl.store(tbase + 4 * K + 4 * fg + kk, tl.where(zm, h3, 0.0), kk < fg)
        it += tl.num_programs(0)
        if ROW_LIST:
            row = tl.load(rows + it, mask=it < batch, other=-1)
        else:
            row = tl.where(tl.load(flags + it) != 0, it, -1)



def gdn_sketch_build_v3(
    state: torch.Tensor,
    flags: torch.Tensor,
    slots: torch.Tensor,
    meta: torch.Tensor,
    sketch: GDNSketchArgs,
    null_block_id: int = NULL_BLOCK_ID,
    rows: torch.Tensor | None = None,
    num_warps: int = 4,
) -> None:
    batch = flags.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    _, hv, v, k = state.shape
    programs = max(1, triton.cdiv(batch, 16))
    _gdn_build_kernel_v3[(programs, hv)](
        state, sketch.u, sketch.phi, sketch.fs, sketch.beta, t.ranks, t.layout,
        slots, meta, flags, flags if rows is None else rows, batch,
        null_block_id, SB=state.stride(0), SH=state.stride(1),
        SV=state.stride(2), SK=state.stride(3), SU=sketch.u.stride(0),
        SM=sketch.phi.stride(0), SF=sketch.fs.stride(0), HV=hv, G=t.rank_cap,
        K=k, V=v, W=t.window, BV=32,
        FS2=triton.next_power_of_2(t.window * t.rank_cap),
        ROW_LIST=rows is not None, num_warps=num_warps,
    )  # fmt: skip


def gdn_sketch_build(
    state: torch.Tensor,
    flags: torch.Tensor,
    slots: torch.Tensor,
    meta: torch.Tensor,
    sketch: GDNSketchArgs,
    null_block_id: int = NULL_BLOCK_ID,
    rows: torch.Tensor | None = None,
    rows_per_program: int = 4,
) -> None:
    """Build the sketch of every flagged row from its state.

    ``rows`` is an optional -1 padded list of the flagged rows.
    """
    batch = flags.shape[0]
    if batch == 0:
        return
    if state.device.type == "npu":
        # The unchunked build kernel overflows triton-ascend's UB planner;
        # the V-chunked variant fits and avoids the torch fallback's ~100
        # small device ops plus a rows.tolist() host sync per call.
        import os
        if os.environ.get("SKETCHSSM_NPU_TORCH_BUILD"):
            if os.environ.get("SKETCHSSM_NPU_DEBUG_SYNC"):
                torch.npu.synchronize()
            if rows is not None:
                rlist = rows.tolist()
                cut = rlist.index(-1) if -1 in rlist else len(rlist)
                row_ids = [r for r in rlist[:cut] if r >= 0]
            else:
                row_ids = [i for i, f in enumerate(flags.tolist()) if f != 0]
            gdn_sketch_build_torch(state, slots, meta, row_ids, sketch, null_block_id)
            return
        if os.environ.get("SKETCHSSM_BUILD_V3", "0") == "1":
            gdn_sketch_build_v3(state, flags, slots, meta, sketch,
                                null_block_id, rows=rows)
            return
        gdn_sketch_build_v2(state, flags, slots, meta, sketch, null_block_id,
                            rows=rows)
        return
    t = sketch.tables
    _, hv, v, k = state.shape
    assert (v, k) == (GDN_SKETCH_HEAD_DIM, GDN_SKETCH_HEAD_DIM)
    assert slots.dtype == torch.int32 and meta.dtype == torch.int32
    programs = batch if rows is None else triton.cdiv(batch, rows_per_program)
    build_kwargs = {}
    if current_platform.device_type == "npu":
        build_kwargs.update(num_stages=1, multibuffer=False,
                            enable_ubuf_saving=True)
    _gdn_build_kernel[(programs, hv)](
        state, sketch.u, sketch.phi, sketch.fs, sketch.beta, t.ranks, t.layout,
        slots, meta, flags, flags if rows is None else rows, batch, null_block_id,
        SB=state.stride(0), SH=state.stride(1), SV=state.stride(2),
        SK=state.stride(3), SU=sketch.u.stride(0), SM=sketch.phi.stride(0),
        SF=sketch.fs.stride(0), HV=hv, G=t.rank_cap, K=k, V=v,
        W=t.window, ROW_LIST=rows is not None, num_warps=4,
        **build_kwargs,
    )  # fmt: skip


def gdn_sketch_build_torch(
    state: torch.Tensor,
    slots: torch.Tensor,
    meta: torch.Tensor,
    row_ids: list[int],
    sketch: "GDNSketchArgs",
    null_block_id: int = NULL_BLOCK_ID,
    row_chunk: int = 32,
) -> None:
    """Torch fallback of ``_gdn_build_kernel`` for non-CUDA platforms.

    Vectorised over (row, head); mirrors the kernel exactly, including the
    BF16 casts of ``u``, ``phi`` and ``fs`` and the packed layout from
    ``gdn_sketch_layout``.
    """
    t = sketch.tables
    hv, v_dim, k_dim = state.shape[1], state.shape[2], state.shape[3]
    window = t.window
    rank_cap = t.rank_cap
    ranks = t.ranks.to(device=state.device, dtype=torch.int64)
    layout = t.layout.to(device=state.device, dtype=torch.int64)
    u, phi, fs, beta = sketch.u, sketch.phi, sketch.fs, sketch.beta
    device = state.device
    cols = torch.arange(k_dim, device=device)
    vg = torch.arange(v_dim, device=device)
    su = sketch.u.stride(0)
    sm = sketch.phi.stride(0)

    for lo in range(0, len(row_ids), row_chunk):
        chunk = row_ids[lo:lo + row_chunk]
        slot_l = [int(slots[i]) for i in chunk]
        keep = [i for i, sl in enumerate(slot_l) if sl > null_block_id and sl >= 0]
        if not keep:
            continue
        rows = [chunk[i] for i in keep]
        n = len(rows)
        slot_t = torch.tensor([slot_l[i] for i in keep], device=device,
                              dtype=torch.int64)
        cidx_t = torch.tensor([int(meta[i]) for i in keep], device=device,
                              dtype=torch.int64)

        # beta zero for every head of every kept row.
        beta[cidx_t] = 0

        act = torch.nonzero(ranks > 0, as_tuple=False).flatten()  # (HA,)
        ha = act.numel()
        if ha == 0:
            continue
        m_a = ranks[act]                                     # (HA,)
        u_off_a, m_off_a, f_off_a, fg_a = layout[act].unbind(-1)

        st = state[slot_t][:, act]                           # (n, HA, V, K)

        # 1. zero each active head's fs block (W * fg entries).
        fg_max = int(fg_a.max())
        wa = torch.arange(window, device=device)
        cc = torch.arange(fg_max, device=device)
        fs_idx = (cidx_t[:, None, None, None] * sketch.fs.stride(0)
                  + f_off_a[None, :, None, None]
                  + wa[None, None, :, None] * fg_a[None, :, None, None]
                  + cc[None, None, None, :])
        fs_mask = (cc[None, None, None, :] < fg_a[None, :, None, None]).expand(
            len(keep), ha, window, fg_max)
        fs.view(-1)[fs_idx[fs_mask]] = 0

        # 2. store u: transposed state columns, j < min(m, G).
        jg = torch.arange(k_dim, device=device)
        u_idx = (cidx_t[:, None, None, None] * su
                 + u_off_a[None, :, None, None]
                 + jg[None, None, None, :] * v_dim
                 + vg[None, None, :, None])
        u_mask = (jg[None, None, None, :] < m_a[None, :, None, None]) & (
            jg[None, None, None, :] < rank_cap)
        # u element (v, j) lives at u_off + j*V + v; u_idx dims (n, HA, V, K)
        u_mask = u_mask.expand(len(keep), ha, v_dim, k_dim)
        # already match st[r, h, v, j].
        sel_idx = u_idx[u_mask]
        u.view(-1)[sel_idx] = st[u_mask].to(u.dtype)

        # 3. Gram-Schmidt for 0 < m < K heads.
        mid = (m_a > 0) & (m_a < k_dim)                      # (HA,)
        if not mid.any():
            continue
        hm = int(mid.sum())
        m_h = m_a[mid]                                       # (HM,)
        moff_h = m_off_a[mid]
        fg_h = fg_a[mid]
        st_m = st[:, mid]                                    # (n, HM, V, K)
        sflat = st_m.reshape(n * hm, v_dim, k_dim)           # (B, V, K)
        m_flat = m_h.repeat(n)                               # (B,)
        moff_flat = moff_h.repeat(n)
        fg_flat = fg_h.repeat(n)
        cid_flat = cidx_t.repeat_interleave(hm)              # (B,)

        energy = sflat.square().sum(1)                       # (B, K)
        mean = energy.sum(-1, keepdim=True) / k_dim
        safe_mean = torch.where(mean > 0, mean, torch.ones_like(mean))
        smrt = safe_mean.sqrt()

        def pivot(idx, prev):
            col = sflat[:, :, idx]                           # (B, V)
            w = col
            for _ in range(2):
                for pv in prev:
                    w = w - pv * (pv * w).sum(-1, keepdim=True)
            e = w.square().sum(-1)
            base_e = col.square().sum(-1)
            keep_m = e > (1.0e-12 * base_e if idx > 0 else torch.zeros_like(e))
            active = m_flat > idx
            ok = (keep_m & active).unsqueeze(-1)
            return torch.where(ok, w / torch.sqrt(
                torch.where(keep_m & active, e, torch.ones_like(e)).unsqueeze(-1)),
                0.0)

        v0 = pivot(0, [])
        v1 = pivot(1, [v0])
        v2 = pivot(2, [v0, v1])
        v3 = pivot(3, [v0, v1, v2])
        z0 = (v0[:, :, None] * sflat).sum(1) / smrt
        z1 = (v1[:, :, None] * sflat).sum(1) / smrt
        z2 = (v2[:, :, None] * sflat).sum(1) / smrt
        z3 = (v3[:, :, None] * sflat).sum(1) / smrt

        residual = (energy / safe_mean - z0.square() - z1.square()
                    - z2.square() - z3.square()).clamp_min(0.0)
        jmask = cols[None, :] < torch.minimum(
            m_flat, torch.full_like(m_flat, 4)).unsqueeze(-1)
        residual = torch.where(jmask, 0.0, residual)
        denom = residual + 0.1
        a_vec = residual / denom
        b0 = torch.where(cols[None, :] < m_flat[:, None], z0 / denom, 0.0)
        b1 = torch.where(cols[None, :] < m_flat[:, None], z1 / denom, 0.0)
        b2 = torch.where(cols[None, :] < m_flat[:, None], z2 / denom, 0.0)
        b3 = torch.where(cols[None, :] < m_flat[:, None], z3 / denom, 0.0)
        l0_0 = (1.0 + (z0 * b0).sum(-1)).sqrt()
        l1_0 = (z1 * b0).sum(-1) / l0_0
        l1_1 = (1.0 + (z1 * b1).sum(-1) - l1_0 * l1_0).sqrt()
        l2_0 = (z2 * b0).sum(-1) / l0_0
        l2_1 = ((z2 * b1).sum(-1) - l2_0 * l1_0) / l1_1
        l2_2 = (1.0 + (z2 * b2).sum(-1) - l2_0 * l2_0 - l2_1 * l2_1).sqrt()
        l3_0 = (z3 * b0).sum(-1) / l0_0
        l3_1 = ((z3 * b1).sum(-1) - l3_0 * l1_0) / l1_1
        l3_2 = ((z3 * b2).sum(-1) - l3_0 * l2_0 - l3_1 * l2_1) / l2_2
        l3_3 = (1.0 + (z3 * b3).sum(-1) - l3_0 * l3_0 - l3_1 * l3_1
                - l3_2 * l3_2).sqrt()
        y0 = b0 / l0_0.unsqueeze(-1)
        y1 = (b1 - l1_0.unsqueeze(-1) * y0) / l1_1.unsqueeze(-1)
        y2 = (b2 - l2_0.unsqueeze(-1) * y0 - l2_1.unsqueeze(-1) * y1) / l2_2.unsqueeze(-1)
        y3 = (b3 - l3_0.unsqueeze(-1) * y0 - l3_1.unsqueeze(-1) * y1
              - l3_2.unsqueeze(-1) * y2) / l3_3.unsqueeze(-1)
        g3 = y3 / l3_3.unsqueeze(-1)
        g2 = (y2 - l3_2.unsqueeze(-1) * g3) / l2_2.unsqueeze(-1)
        g1 = (y1 - l2_1.unsqueeze(-1) * g2 - l3_1.unsqueeze(-1) * g3) / l1_1.unsqueeze(-1)
        g0 = (y0 - l1_0.unsqueeze(-1) * g1 - l2_0.unsqueeze(-1) * g2
              - l3_0.unsqueeze(-1) * g3) / l0_0.unsqueeze(-1)
        factor = torch.where(cols[None, :] < m_flat[:, None], 0.1 / denom,
                             torch.ones_like(denom))

        # merged maps for m <= 4 rows: merged_n = sum_j g_j[n] * z_j * factor
        for nq in range(4):
            sel = (m_flat > nq) & (m_flat <= 4)
            if not sel.any():
                continue
            merged = (g0[sel].select(1, nq).unsqueeze(-1) * z0[sel]
                      + g1[sel].select(1, nq).unsqueeze(-1) * z1[sel]
                      + g2[sel].select(1, nq).unsqueeze(-1) * z2[sel]
                      + g3[sel].select(1, nq).unsqueeze(-1) * z3[sel]) * factor[sel]
            dst = (cid_flat[sel] * sm + moff_flat[sel] + nq * k_dim
                   ).unsqueeze(-1) + cols[None, :]
            phi.view(-1)[dst] = merged.to(phi.dtype)

        # general rows (4 < m < K): z blocks (unmasked), a and g blocks.
        big = m_flat > 4
        if big.any():
            for jdx, zb in enumerate((z0, z1, z2, z3)):
                dst = (cid_flat[big] * sm + moff_flat[big]
                       + jdx * k_dim).unsqueeze(-1) + cols[None, :]
                phi.view(-1)[dst] = (zb[big] * factor[big]).to(phi.dtype)
            base = (cid_flat[big] * sm + moff_flat[big]
                    + 4 * k_dim).unsqueeze(-1) + cols[None, :]
            inblk = cols[None, :] < fg_flat[big][:, None]
            vals = torch.where(cols[None, :] < m_flat[big][:, None],
                               a_vec[big], 0.0)
            phi.view(-1)[base[inblk]] = vals[inblk].to(phi.dtype)
            for qdx, gb in enumerate((g0, g1, g2, g3)):
                dst = (cid_flat[big] * sm + moff_flat[big] + 4 * k_dim
                       + (qdx + 1) * fg_flat[big]).unsqueeze(-1) + cols[None, :]
                vals = torch.where(cols[None, :] < m_flat[big][:, None],
                                   gb[big], 0.0)
                phi.view(-1)[dst[inblk]] = vals[inblk].to(phi.dtype)
