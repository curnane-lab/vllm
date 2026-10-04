# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM Gated DeltaNet decode kernels (Triton)."""

import os

import torch

from vllm.model_executor.layers.mamba.ops.gdn_sketchssm_common import (
    GDN_SKETCH_HEAD_DIM,
    GDN_SKETCH_PIVOTS,
    GDNSketchArgs,
    gdn_sketch_build,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

# Flush rows per program of the flush and rebuild launches.
TRITON_ROWS_PER_PROGRAM = 4

# Persistent two-phase scratch buffers keyed by (batch, hv, cap, device).
_scratch_cache: dict = {}


@triton.jit
def _gdn_sketch_step_kernel(
    qkv, a_act, b_act, A_log, dt_bias, out, state, d_cache, k_cache, g_cache,
    slots, write_pos, meta, u, phi, fs, beta_ring, current_d, current_k, ranks,
    layout, scale, s_qkv, s_a, s_b, s_st_slot, s_st_head, s_st_v, s_d_slot,
    s_d_head, s_d_pos, s_k_slot, s_k_head, s_k_pos, s_g_slot, s_g_head, s_u,
    s_phi, s_fs, s_beta, s_cd, s_ck, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, W: tl.constexpr, WP: tl.constexpr,
    P: tl.constexpr, BG: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    # One (row, value head). Window rows are padded to WP (a power of two).
    n = tl.program_id(0)
    hv = tl.program_id(1)
    i_h = hv // (HV // H)
    kk = tl.arange(0, K)
    vv = tl.arange(0, V)
    ww = tl.arange(0, WP)
    pp = tl.arange(0, P)
    slot = tl.load(slots + n).to(tl.int64)
    p_o = out + (n * HV + hv) * V + vv
    if slot <= 0:
        tl.store(p_o, tl.zeros([V], tl.float32).to(p_o.dtype.element_ty))
    else:
        wp = tl.load(write_pos + n)
        cidx = tl.load(meta + n).to(tl.int64)
        m = tl.load(ranks + hv)
        p_q = qkv + n * s_qkv + i_h * K
        p_k = p_q + H * K
        q = tl.load(p_q + kk).to(tl.float32)
        k = tl.load(p_k + kk).to(tl.float32)
        v = tl.load(qkv + n * s_qkv + 2 * H * K + hv * V + vv).to(tl.float32)
        a_val = tl.load(a_act + n * s_a + hv).to(tl.float32)
        b_val = tl.load(b_act + n * s_b + hv).to(tl.float32)
        xg = a_val + tl.load(dt_bias + hv).to(tl.float32)
        sp = tl.where(xg <= 20.0, tl.log(1.0 + tl.exp(xg)), xg)
        g_val = -tl.exp(tl.load(A_log + hv).to(tl.float32)) * sp
        alpha = tl.exp(g_val)
        beta = 1.0 / (1.0 + tl.exp(-b_val))
        beta = beta.to(b_act.dtype.element_ty).to(tl.float32)
        tl.store(beta_ring + cidx * s_beta + hv * W + wp, beta)
        q_sc = 1.0 / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
        k_rn = 1.0 / tl.sqrt(tl.sum(k * k) + 1e-6)
        qn = q * q_sc
        kn = k * k_rn
        cur_kq = tl.sum(qn * kn)

        # Ring replay weights over the window rows s < wp.
        valid = ww < wp
        k_ring = k_cache + slot * s_k_slot + i_h * s_k_head
        keys = tl.load(
            k_ring + ww[:, None] * s_k_pos + kk[None, :], mask=valid[:, None], other=0.0
        ).to(tl.float32)
        kq_s = tl.sum(keys * qn[None, :], axis=1)
        kk_s = tl.sum(keys * kn[None, :], axis=1)
        g_ring = g_cache + slot * s_g_slot + hv * s_g_head
        gs = tl.load(g_ring + ww, mask=valid, other=0.0)
        pre = tl.cumsum(gs, axis=0)
        gtot = tl.sum(gs)
        rep = tl.where(valid, tl.exp(gtot - pre), 0.0)
        tot = tl.exp(gtot)
        d_ring = d_cache + slot * s_d_slot + hv * s_d_head
        ds = tl.load(
            d_ring + ww[:, None] * s_d_pos + vv[None, :], mask=valid[:, None], other=0.0
        ).to(tl.float32)
        s_q = tl.sum(ds * (kq_s * rep)[:, None], axis=0)
        s_k = tl.sum(ds * (kk_s * rep)[:, None], axis=0)

        hq = tl.zeros([V], tl.float32)
        hk = tl.zeros([V], tl.float32)
        if m == 0:
            p_s = state + slot * s_st_slot + hv * s_st_head + vv[:, None] * s_st_v
            for k0 in range(0, K, BK):
                kc = k0 + tl.arange(0, BK)
                s = tl.load(p_s + kc[None, :])
                qc = tl.load(p_q + kc).to(tl.float32) * q_sc
                kcv = tl.load(p_k + kc).to(tl.float32) * k_rn
                hq += tl.sum(s * qc[None, :], axis=1)
                hk += tl.sum(s * kcv[None, :], axis=1)
        else:
            u_off = tl.load(layout + hv * 4)
            phi_off = tl.load(layout + hv * 4 + 1)
            fs_off = tl.load(layout + hv * 4 + 2)
            fg = tl.load(layout + hv * 4 + 3)
            p_phi = phi + cidx * s_phi + phi_off
            p_fs = fs + cidx * s_fs + fs_off
            p_u = u + cidx * s_u + u_off
            # Pivot rows (or the merged maps for m <= 4) dotted with q and k.
            piv = tl.load(
                p_phi + pp[:, None] * K + kk[None, :],
                mask=(pp[:, None] < tl.minimum(m, P)) & (m < K),
                other=0.0,
            ).to(tl.float32)
            tq = tl.sum(piv * qn[None, :], axis=1)
            tk = tl.sum(piv * kn[None, :], axis=1)
            mid = (m > P) & (m < K)
            for g0 in range(0, m, BG):
                g = g0 + tl.arange(0, BG)
                gm = g < m
                qg = tl.load(p_q + g, mask=gm, other=0.0).to(tl.float32) * q_sc
                kg = tl.load(p_k + g, mask=gm, other=0.0).to(tl.float32) * k_rn
                sel = g[:, None] == pp[None, :]
                xq = tl.sum(tl.where(sel, tq[None, :], 0.0), axis=1)
                xk = tl.sum(tl.where(sel, tk[None, :], 0.0), axis=1)
                ad = tl.load(p_phi + P * K + g, mask=gm & mid, other=0.0)
                gains = tl.load(
                    p_phi + P * K + (pp[:, None] + 1) * fg + g[None, :],
                    mask=gm[None, :] & mid,
                    other=0.0,
                ).to(tl.float32)
                xq_mid = ad.to(tl.float32) * qg + tl.sum(gains * tq[:, None], axis=0)
                xk_mid = ad.to(tl.float32) * kg + tl.sum(gains * tk[:, None], axis=0)
                xq = tl.where(m <= P, xq, tl.where(m < K, xq_mid, qg))
                xk = tl.where(m <= P, xk, tl.where(m < K, xk_mid, kg))
                f = tl.load(
                    p_fs + ww[:, None] * fg + g[None, :],
                    mask=valid[:, None] & gm[None, :],
                    other=0.0,
                ).to(tl.float32)
                eq = tl.sum(f * kq_s[:, None], axis=0)
                ek = tl.sum(f * kk_s[:, None], axis=0)
                fcur = beta * (xk - ek)
                x = xq - eq - fcur * cur_kq
                tl.store(p_fs + wp * fg + g, fcur.to(fs.dtype.element_ty), mask=gm)
                us = tl.load(
                    p_u + g[:, None] * V + vv[None, :], mask=gm[:, None], other=0.0
                ).to(tl.float32)
                hq += tl.sum(us * x[:, None], axis=0)

        stq = alpha * (hq * tot + s_q)
        stk = alpha * (hk * tot + s_k)
        dc = tl.where(m > 0, beta * (v - alpha * s_k), beta * (v - stk))
        tl.store(p_o, (stq + dc * cur_kq).to(p_o.dtype.element_ty))
        tl.store(g_ring + wp, g_val)
        if wp == W - 1:
            tl.store(current_d + cidx * s_cd + hv * V + vv, dc)
        else:
            tl.store(d_ring + wp * s_d_pos + vv, dc.to(d_cache.dtype.element_ty))
        if hv % (HV // H) == 0:
            if wp == W - 1:
                tl.store(current_k + cidx * s_ck + i_h * K + kk, kn)
            else:
                tl.store(k_ring + wp * s_k_pos + kk, kn.to(k_cache.dtype.element_ty))


# Two-phase step decode: K1 computes the per-g scalars and ring projections,
# K2 is a pure V-local pass (state/u reads and out writes) split across
# programs. Scratch row layout (fp32): [alpha, tot, beta, cur_kq, q_sc, k_rn,
# pad, pad | x_g (CAP) | s_q (V) | s_k (V) | dc (V)].
_SCR_SCALARS = 8


@triton.jit
def _gdn_sketch_step_k1_kernel(
    qkv, a_act, b_act, A_log, dt_bias, d_cache, k_cache, g_cache, slots,
    write_pos, meta, phi, fs, beta_ring, current_d, current_k, ranks, layout,
    scratch, scale, s_qkv, s_a, s_b, s_d_slot, s_d_head, s_d_pos, s_k_slot,
    s_k_head, s_k_pos, s_g_slot, s_g_head, s_phi, s_fs, s_beta, s_cd, s_ck,
    s_sc, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    V: tl.constexpr, W: tl.constexpr, WP: tl.constexpr, P: tl.constexpr,
    CAP: tl.constexpr,
):  # fmt: skip
    # One (row, value head): per-g scalars x_g, ring weights and projections.
    n = tl.program_id(0)
    hv = tl.program_id(1)
    i_h = hv // (HV // H)
    kk = tl.arange(0, K)
    vv = tl.arange(0, V)
    ww = tl.arange(0, WP)
    pp = tl.arange(0, P)
    slot = tl.load(slots + n).to(tl.int64)
    if slot <= 0:
        return
    wp = tl.load(write_pos + n)
    cidx = tl.load(meta + n).to(tl.int64)
    m = tl.load(ranks + hv)
    p_sc = scratch + (n * HV + hv) * s_sc
    p_q = qkv + n * s_qkv + i_h * K
    p_k = p_q + H * K
    q = tl.load(p_q + kk).to(tl.float32)
    k = tl.load(p_k + kk).to(tl.float32)
    a_val = tl.load(a_act + n * s_a + hv).to(tl.float32)
    b_val = tl.load(b_act + n * s_b + hv).to(tl.float32)
    xg = a_val + tl.load(dt_bias + hv).to(tl.float32)
    spg = tl.where(xg <= 20.0, tl.log(1.0 + tl.exp(xg)), xg)
    g_val = -tl.exp(tl.load(A_log + hv).to(tl.float32)) * spg
    alpha = tl.exp(g_val)
    beta = 1.0 / (1.0 + tl.exp(-b_val))
    beta = beta.to(b_act.dtype.element_ty).to(tl.float32)
    tl.store(beta_ring + cidx * s_beta + hv * W + wp, beta)
    q_sc = 1.0 / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
    k_rn = 1.0 / tl.sqrt(tl.sum(k * k) + 1e-6)
    qn = q * q_sc
    kn = k * k_rn
    cur_kq = tl.sum(qn * kn)

    # Ring replay weights over the window rows s < wp.
    valid = ww < wp
    k_ring = k_cache + slot * s_k_slot + i_h * s_k_head
    keys = tl.load(
        k_ring + ww[:, None] * s_k_pos + kk[None, :], mask=valid[:, None], other=0.0
    ).to(tl.float32)
    kq_s = tl.sum(keys * qn[None, :], axis=1)
    kk_s = tl.sum(keys * kn[None, :], axis=1)
    g_ring = g_cache + slot * s_g_slot + hv * s_g_head
    gs = tl.load(g_ring + ww, mask=valid, other=0.0)
    pre = tl.cumsum(gs, axis=0)
    gtot = tl.sum(gs)
    rep = tl.where(valid, tl.exp(gtot - pre), 0.0)
    tot = tl.exp(gtot)
    d_ring = d_cache + slot * s_d_slot + hv * s_d_head
    ds = tl.load(
        d_ring + ww[:, None] * s_d_pos + vv[None, :], mask=valid[:, None], other=0.0
    ).to(tl.float32)
    s_q = tl.sum(ds * (kq_s * rep)[:, None], axis=0)
    s_k = tl.sum(ds * (kk_s * rep)[:, None], axis=0)

    tl.store(p_sc + 0, alpha)
    tl.store(p_sc + 1, tot)
    tl.store(p_sc + 2, beta)
    tl.store(p_sc + 3, cur_kq)
    tl.store(p_sc + 4, q_sc)
    tl.store(p_sc + 5, k_rn)
    tl.store(p_sc + 8 + CAP + vv, s_q)
    tl.store(p_sc + 8 + CAP + V + vv, s_k)
    if m > 0:
        v = tl.load(qkv + n * s_qkv + 2 * H * K + hv * V + vv).to(tl.float32)
        dc = beta * (v - alpha * s_k)
        tl.store(p_sc + 8 + CAP + 2 * V + vv, dc)
        if wp == W - 1:
            tl.store(current_d + cidx * s_cd + hv * V + vv, dc)
        else:
            tl.store(d_ring + wp * s_d_pos + vv, dc.to(d_cache.dtype.element_ty))
        phi_off = tl.load(layout + hv * 4 + 1)
        fs_off = tl.load(layout + hv * 4 + 2)
        fg = tl.load(layout + hv * 4 + 3)
        p_phi = phi + cidx * s_phi + phi_off
        p_fs = fs + cidx * s_fs + fs_off
        # Loop-free per-g scalars on a CAP-wide tile (m <= CAP < K here;
        # rank_cap bounds m). The m <= P heads use CAP-resolved pivot rows,
        # avoiding a one-hot gather matrix.
        gg = tl.arange(0, CAP)
        gmv = gg < m
        if m <= P:
            piv = tl.load(
                p_phi + gg[:, None] * K + kk[None, :],
                mask=(gg[:, None] < tl.minimum(m, P)) & (m < K),
                other=0.0,
            ).to(tl.float32)
            xq = tl.sum(piv * qn[None, :], axis=1)
            xk = tl.sum(piv * kn[None, :], axis=1)
        else:
            pivs = tl.load(
                p_phi + pp[:, None] * K + kk[None, :],
                mask=(pp[:, None] < tl.minimum(m, P)) & (m < K),
                other=0.0,
            ).to(tl.float32)
            tq = tl.sum(pivs * qn[None, :], axis=1)
            tk = tl.sum(pivs * kn[None, :], axis=1)
            qg = tl.load(p_q + gg, mask=gmv, other=0.0).to(tl.float32) * q_sc
            kg = tl.load(p_k + gg, mask=gmv, other=0.0).to(tl.float32) * k_rn
            mid = (m > P) & (m < K)
            ad = tl.load(p_phi + P * K + gg, mask=gmv & mid,
                         other=0.0).to(tl.float32)
            gains = tl.load(
                p_phi + P * K + (pp[:, None] + 1) * fg + gg[None, :],
                mask=gmv[None, :] & mid,
                other=0.0,
            ).to(tl.float32)
            gtq = tl.sum(gains * tq[:, None], axis=0)
            gtk = tl.sum(gains * tk[:, None], axis=0)
            xq = tl.where(m < K, ad * qg + gtq, qg)
            xk = tl.where(m < K, ad * kg + gtk, kg)
        f = tl.load(
            p_fs + ww[:, None] * fg + gg[None, :],
            mask=valid[:, None] & gmv[None, :],
            other=0.0,
        ).to(tl.float32)
        ek = tl.sum(f * kk_s[:, None], axis=0)
        eq = tl.sum(f * kq_s[:, None], axis=0)
        fcur = beta * (xk - ek)
        x = xq - eq - fcur * cur_kq
        tl.store(p_fs + wp * fg + gg, fcur.to(fs.dtype.element_ty), mask=gmv)
        tl.store(p_sc + 8 + gg, x, mask=gmv)
    tl.store(g_ring + wp, g_val)
    if hv % (HV // H) == 0:
        if wp == W - 1:
            tl.store(current_k + cidx * s_ck + i_h * K + kk, kn)
        else:
            tl.store(k_ring + wp * s_k_pos + kk, kn.to(k_cache.dtype.element_ty))


@triton.jit
def _gdn_sketch_step_k2_kernel(
    qkv, out, state, d_cache, slots, write_pos, meta, u, current_d, ranks,
    layout, scratch, s_qkv, s_st_slot, s_st_head, s_st_v, s_d_slot, s_d_head,
    s_d_pos, s_u, s_cd, s_sc, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, W: tl.constexpr, WP: tl.constexpr,
    BV: tl.constexpr, CAP: tl.constexpr,
):  # fmt: skip
    # Pure V-local output pass: hq from the state (m == 0) or from u @ x_g,
    # then out. The dc/d_ring stores for m == 0 also live here.
    n = tl.program_id(0)
    hv = tl.program_id(1)
    vb = tl.program_id(2) * BV + tl.arange(0, BV)
    slot = tl.load(slots + n).to(tl.int64)
    p_o = out + (n * HV + hv) * V + vb
    if slot <= 0:
        tl.store(p_o, tl.zeros([BV], tl.float32).to(p_o.dtype.element_ty))
        return
    wp = tl.load(write_pos + n)
    m = tl.load(ranks + hv)
    p_sc = scratch + (n * HV + hv) * s_sc
    alpha = tl.load(p_sc + 0)
    tot = tl.load(p_sc + 1)
    cur_kq = tl.load(p_sc + 3)
    s_q = tl.load(p_sc + 8 + CAP + vb)
    i_h = hv // (HV // H)
    if m == 0:
        kk = tl.arange(0, K)
        q_sc = tl.load(p_sc + 4)
        k_rn = tl.load(p_sc + 5)
        q = tl.load(qkv + n * s_qkv + i_h * K + kk).to(tl.float32) * q_sc
        k = tl.load(qkv + n * s_qkv + H * K + i_h * K + kk).to(tl.float32) * k_rn
        s = tl.load(
            state + slot * s_st_slot + hv * s_st_head + vb[:, None] * s_st_v
            + kk[None, :]
        )
        hq = tl.sum(s * q[None, :], axis=1)
        hk = tl.sum(s * k[None, :], axis=1)
        beta = tl.load(p_sc + 2)
        s_k = tl.load(p_sc + 8 + CAP + V + vb)
        stk = alpha * (hk * tot + s_k)
        v = tl.load(qkv + n * s_qkv + 2 * H * K + hv * V + vb).to(tl.float32)
        dc = beta * (v - stk)
        if wp == W - 1:
            tl.store(current_d + tl.load(meta + n).to(tl.int64) * s_cd + hv * V + vb, dc)
        else:
            d_ring = d_cache + slot * s_d_slot + hv * s_d_head
            tl.store(d_ring + wp * s_d_pos + vb, dc.to(d_cache.dtype.element_ty))
        o = alpha * (hq * tot + s_q) + dc * cur_kq
    else:
        g = tl.arange(0, CAP)
        u_off = tl.load(layout + hv * 4)
        us = tl.load(
            u + tl.load(meta + n).to(tl.int64) * s_u + u_off
            + g[:, None] * V + vb[None, :],
            mask=(g < m)[:, None],
            other=0.0,
        ).to(tl.float32)
        x = tl.load(p_sc + 8 + g, mask=g < m, other=0.0)
        hq = tl.sum(us * x[:, None], axis=0)
        dc = tl.load(p_sc + 8 + CAP + 2 * V + vb)
        o = alpha * (hq * tot + s_q) + dc * cur_kq
    tl.store(p_o, o.to(p_o.dtype.element_ty))


@triton.jit
def _gdn_sketch_flush_kernel(
    qkv, out, state, d_cache, k_cache, g_cache, slots, meta, flush_rows, batch,
    beta_ring, current_d, current_k, ranks, scale, s_qkv, s_st_slot,
    s_st_head, s_st_v, s_d_slot, s_d_head, s_d_pos, s_k_slot, s_k_head,
    s_k_pos, s_g_slot, s_g_head, s_beta, s_cd, s_ck, H: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, W: tl.constexpr,
    WP: tl.constexpr, BV: tl.constexpr, LOGW: tl.constexpr,
    NEUMAN: tl.constexpr, PROJ_PRECISION: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):  # fmt: skip
    # Flush of one value head of the rows in flush_rows (-1 padded). The
    # stored updates lack the window-start state's erase, which the WY form
    # restores: S' = tot S + X K with X = D rep - (S K^T) T.
    # Window rows are padded to WP with zero keys, updates and gates.
    hv = tl.program_id(1)
    i_h = hv // (HV // H)
    kk = tl.arange(0, K)
    ww = tl.arange(0, WP)
    inw = ww < W
    it = tl.program_id(0)
    row = tl.load(flush_rows + it, mask=it < batch, other=-1)
    while row >= 0:
        slot = tl.load(slots + row).to(tl.int64)
        if slot > 0:
            cidx = tl.load(meta + row).to(tl.int64)
            m = tl.load(ranks + hv)
            last = ww == W - 1
            ring = inw & ~last
            keys = tl.load(
                k_cache + slot * s_k_slot + i_h * s_k_head + ww[:, None] * s_k_pos
                + kk[None, :],
                mask=ring[:, None],
                other=0.0,
            ).to(tl.float32)  # fmt: skip
            k_cur = tl.load(current_k + cidx * s_ck + i_h * K + kk)
            k_cur = k_cur.to(tl.bfloat16).to(tl.float32)
            keys = tl.where(last[:, None], k_cur[None, :], keys)
            gs = tl.load(
                g_cache + slot * s_g_slot + hv * s_g_head + ww, mask=inw, other=0.0
            )
            pre = tl.cumsum(gs, axis=0)
            gt = tl.sum(gs)
            rep = tl.exp(gt - pre)
            tot = tl.exp(gt)
            q = tl.load(qkv + row * s_qkv + i_h * K + kk).to(tl.float32)
            qn = q * (1.0 / tl.sqrt(tl.sum(q * q) + 1e-6) * scale)
            tm = tl.zeros([WP, WP], tl.float32)
            if m > 0:
                bs = tl.load(
                    beta_ring + cidx * s_beta + hv * W + ww, mask=inw, other=0.0
                )
                gram = tl.dot(keys, tl.trans(keys), input_precision=PROJ_PRECISION)
                upper = ww[:, None] < ww[None, :]
                gamma = gram * bs[None, :] * tl.exp(pre[None, :] - pre[:, None])
                gamma = tl.where(upper, gamma, 0.0)
                inv = tl.zeros([WP, WP], tl.float32)
                if NEUMAN:
                    # (I + gamma)^-1 = sum (−gamma)^k for a strictly upper
                    # nilpotent gamma; log-doubling needs LOGW dot rounds.
                    neg = -gamma
                    pa = neg
                    qa = neg
                    for _ in tl.static_range(LOGW):
                        pa = pa + tl.dot(pa, qa, input_precision=DOT_PRECISION)
                        qa = tl.dot(qa, qa, input_precision=DOT_PRECISION)
                    inv = pa + tl.where(ww[:, None] == ww[None, :], 1.0, 0.0)
                if W == 16 and not NEUMAN:
                    for i in tl.static_range(W):
                        r = W - 1 - i
                        grow = tl.sum(tl.where(ww[:, None] == r, gamma, 0.0), axis=0)
                        new = tl.where(ww == r, 1.0, 0.0)
                        new -= tl.sum(grow[:, None] * inv, axis=0)
                        inv = tl.where(ww[:, None] == r, new[None, :], inv)
                else:
                    for i in range(W):
                        r = W - 1 - i
                        grow = tl.sum(tl.where(ww[:, None] == r, gamma, 0.0), axis=0)
                        new = tl.where(ww == r, 1.0, 0.0)
                        new -= tl.sum(grow[:, None] * inv, axis=0)
                        inv = tl.where(ww[:, None] == r, new[None, :], inv)
                tm = inv * bs[:, None] * tl.exp(pre[:, None] + gt - pre[None, :])
                tm = tl.where(ww[:, None] <= ww[None, :], tm, 0.0)
            p_s = state + slot * s_st_slot + hv * s_st_head
            d_ring = d_cache + slot * s_d_slot + hv * s_d_head
            for v0 in range(0, V, BV):
                vb = v0 + tl.arange(0, BV)
                ptr = p_s + vb[:, None] * s_st_v + kk[None, :]
                s = tl.load(ptr)
                d = tl.load(
                    d_ring + ww[None, :] * s_d_pos + vb[:, None],
                    mask=ring[None, :],
                    other=0.0,
                ).to(tl.float32)
                d_cur = tl.load(current_d + cidx * s_cd + hv * V + vb)
                x = tl.where(last[None, :], d_cur[:, None], d) * rep[None, :]
                if m > 0:
                    proj = tl.dot(s, tl.trans(keys), input_precision=PROJ_PRECISION)
                    x -= tl.dot(proj, tm, input_precision=DOT_PRECISION)
                s = s * tot + tl.dot(x, keys, input_precision=DOT_PRECISION)
                tl.store(ptr, s)
                o = tl.sum(s * qn[None, :], axis=1)
                tl.store(out + (row * HV + hv) * V + vb, o.to(out.dtype.element_ty))
        it += tl.num_programs(0)
        row = tl.load(flush_rows + it, mask=it < batch, other=-1)


def gdn_sketch_triton_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    slots: torch.Tensor,
    write_pos: torch.Tensor,
    meta: torch.Tensor,
    flush_rows: torch.Tensor,
    sketch: GDNSketchArgs,
    scale: float,
    null_block_id: int = NULL_BLOCK_ID,
    has_flush_rows: bool = True,
) -> None:
    """One SketchSSM decode step of a GDN layer."""
    batch = mixed_qkv.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    h, hv = k_cache.shape[1], state.shape[1]
    k = v = GDN_SKETCH_HEAD_DIM
    w = t.window
    wp = triton.next_power_of_2(w)
    assert state.shape[2:] == (v, k) and state.stride(3) == 1
    assert d_cache.shape[2:] == (w, v) and d_cache.stride(3) == 1
    assert k_cache.shape[2:] == (w, k) and k_cache.stride(3) == 1
    assert g_cache.shape[2] == w and g_cache.stride(2) == 1
    assert hv % h == 0 and mixed_qkv.stride(1) == 1
    if slots.dim() == 2:
        slots = slots[:, 0]
    # Slots <= 0 are padding.
    assert null_block_id == 0
    assert out.is_contiguous() and out.dtype == mixed_qkv.dtype
    s = sketch
    for x in (s.beta, s.current_d, s.current_k):
        assert x[0].is_contiguous()
    strides = (
        *state.stride()[:3], *d_cache.stride()[:3], *k_cache.stride()[:3],
        *g_cache.stride()[:2], s.u.stride(0),
    )  # fmt: skip
    ascend_done = False
    if (os.environ.get("SKETCHSSM_ASCENDC", "0") == "1"
            and current_platform.device_type == "npu"):
        from vllm.model_executor.layers.mamba.ops.gdn_sketch_step_ascend import (
            gdn_sketch_ascend_step,
        )
        ascend_done = gdn_sketch_ascend_step(
            mixed_qkv, a, b, A_log, dt_bias, out, state, d_cache, k_cache,
            g_cache, slots, write_pos, meta, s, scale)
        if ascend_done and not has_flush_rows:
            return

    two_phase = (os.environ.get("SKETCHSSM_TWO_PHASE", "1") == "1"
                 and batch >= 128)
    if ascend_done:
        pass  # step handled by the AscendC kernel; run flush/build below
    elif two_phase:
        cap = triton.next_power_of_2(t.rank_cap)
        # Persistent scratch: one buffer per (batch, hv, cap), reused across
        # layers and steps. Write-before-read within a step, so reuse is safe
        # both in eager and under graph replay (the pointer is baked once).
        cache = _scratch_cache
        key = (batch, hv, cap, mixed_qkv.device)
        scratch = cache.get(key)
        if scratch is None:
            if len(cache) > 8:
                cache.clear()
            scratch = torch.empty(
                (batch, hv, _SCR_SCALARS + cap + 3 * v),
                dtype=torch.float32, device=mixed_qkv.device,
            )
            cache[key] = scratch
        _gdn_sketch_step_k1_kernel[(batch, hv)](
            mixed_qkv, a, b, A_log, dt_bias, d_cache, k_cache, g_cache, slots,
            write_pos, meta, s.phi, s.fs, s.beta, s.current_d, s.current_k,
            t.ranks, t.layout, scratch, scale, mixed_qkv.stride(0), a.stride(0),
            b.stride(0), *strides[3:11], s.phi.stride(0), s.fs.stride(0),
            s.beta.stride(0), s.current_d.stride(0), s.current_k.stride(0),
            scratch.stride(1), H=h, HV=hv, K=k, V=v, W=w, WP=wp,
            P=GDN_SKETCH_PIVOTS, CAP=cap, num_warps=1,
        )  # fmt: skip
        _gdn_sketch_step_k2_kernel[(batch, hv, v // 64)](
            mixed_qkv, out, state, d_cache, slots, write_pos, meta, s.u,
            s.current_d, t.ranks, t.layout, scratch, mixed_qkv.stride(0),
            *strides[:3], *strides[3:6], s.u.stride(0), s.current_d.stride(0),
            scratch.stride(1), H=h, HV=hv, K=k, V=v, W=w, WP=wp, BV=64,
            CAP=cap, num_warps=2,
        )  # fmt: skip
        if not has_flush_rows:
            return
    else:
        _gdn_sketch_step_kernel[(batch, hv)](
            mixed_qkv, a, b, A_log, dt_bias, out, state, d_cache, k_cache,
            g_cache, slots, write_pos, meta, s.u, s.phi, s.fs, s.beta,
            s.current_d, s.current_k, t.ranks, t.layout, scale,
            mixed_qkv.stride(0), a.stride(0), b.stride(0), *strides,
            s.phi.stride(0), s.fs.stride(0), s.beta.stride(0),
            s.current_d.stride(0), s.current_k.stride(0), H=h, HV=hv, K=k, V=v,
            W=w, WP=wp, P=GDN_SKETCH_PIVOTS, BG=8, BK=8,
            num_warps=min(8, wp // 16),
        )  # fmt: skip
    if not has_flush_rows:
        return
    rocm = current_platform.is_rocm()
    npu = current_platform.device_type == "npu"
    # Above 32 rows, FMA dots on 16-row value blocks need less shared memory.
    fma = wp > 32
    # One program per roughly-flushing row: the grid then matches the active
    # work (batch/16 rows flush per step) and each program runs one row.
    programs = max(1, triton.cdiv(batch, 4 if fma else 16))
    _gdn_sketch_flush_kernel[(programs, hv)](
        mixed_qkv, out, state, d_cache, k_cache, g_cache, slots, meta, flush_rows,
        batch, s.beta, s.current_d, s.current_k, t.ranks, scale,
        mixed_qkv.stride(0), *strides[:-1], s.beta.stride(0), s.current_d.stride(0),
        s.current_k.stride(0), H=h, HV=hv, K=k, V=v, W=w, WP=wp,
        BV=16 if fma else 128,
        LOGW=wp.bit_length() - 1,
        NEUMAN=os.environ.get("SKETCHSSM_FLUSH_NEUMAN", "0") == "1",
        PROJ_PRECISION="ieee" if (rocm or npu or not fma) else "tf32",
        DOT_PRECISION="ieee" if (rocm or npu) else "ieee" if fma else "tf32x3",
        num_warps=8,
    )  # fmt: skip
    # Rebuild the flushed rows' sketches from the new state.
    gdn_sketch_build(
        state, flush_rows, slots, meta, sketch, null_block_id, rows=flush_rows,
        rows_per_program=TRITON_ROWS_PER_PROGRAM,
    )  # fmt: skip
