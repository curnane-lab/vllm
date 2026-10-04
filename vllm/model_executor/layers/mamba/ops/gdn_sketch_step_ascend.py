# SPDX-License-Identifier: Apache-2.0
"""AscendC GDN SketchSSM decode step (A3) — ctypes binding wrapper.

Requires the custom opp package (gdn_sketch_step) to be installed and its
set_env sourced; falls back cleanly when the op is unavailable so the
SKETCHSSM_ASCENDC switch can stay on across environments.
"""

import ctypes
import glob

import torch

_B = None
_AVAILABLE = None


def _binding():
    global _B, _AVAILABLE
    if _AVAILABLE is False:
        return None
    if _B is None:
        cands = (glob.glob("/tmp/probe_opp_v*/vendors/custom_transformer/"
                           "op_api/lib/libcust_opapi.so")
                 + glob.glob("/usr/local/Ascend/opp/vendors/*/op_api/lib/"
                             "libcust_opapi.so", recursive=True)
                 + glob.glob("/mnt/workspace/vllm-ascend/vllm_ascend/"
                             "_cann_ops_custom/**/op_api/lib/libcust_opapi.so",
                             recursive=True))
        if not cands:
            _AVAILABLE = False
            return None
        try:
            lib = ctypes.CDLL(cands[0])
        except OSError:
            _AVAILABLE = False
            return None
        import torch_npu  # noqa: F401

        from fla_npu.ops.ascendc import _runtime as rt

        _B = (rt, rt.runtime(), lib)
        lib.aclnnGdnSketchStepGetWorkspaceSize.restype = ctypes.c_int
        lib.aclnnGdnSketchStep.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p
        ]
        lib.aclnnGdnSketchStep.restype = ctypes.c_int
    return _B


def gdn_sketch_ascend_step(
    mixed_qkv, a, b, A_log, dt_bias, out, state, d_cache, k_cache, g_cache,
    slots, write_pos, meta, sketch, scale,
) -> bool:
    """One decode step through the AscendC kernel; returns False when the
    custom op is unavailable (caller should fall back to triton)."""
    bnd = _binding()
    if bnd is None:
        return False
    rt, runtime, lib = bnd
    s = sketch
    if slots.dim() == 2:
        slots = slots[:, 0]
    tensors = (mixed_qkv, a, b, A_log, dt_bias, state, d_cache, k_cache,
               g_cache, slots, write_pos, meta, s.u, s.phi, s.fs, s.beta,
               s.current_d, s.current_k, s.tables.ranks, s.tables.layout)
    attrs = dict(
        scale=scale, batch=mixed_qkv.shape[0], hv=state.shape[1],
        h=k_cache.shape[1], s_qkv=mixed_qkv.stride(0), s_a=a.stride(0),
        s_b=b.stride(0), s_st_slot=state.stride(0), s_st_head=state.stride(1),
        s_dc_slot=d_cache.stride(0), s_dc_head=d_cache.stride(1),
        s_kc_slot=k_cache.stride(0), s_kc_head=k_cache.stride(1),
        s_gc_slot=g_cache.stride(0), s_gc_head=g_cache.stride(1),
        s_su=s.u.stride(0), s_sm=s.phi.stride(0), s_sf=s.fs.stride(0))
    try:
        ctx = rt._CallContext(runtime, out.device)
    except Exception:
        return False
    try:
        args = [ctx.tensor(t) for t in tensors]
        args.append(ctypes.c_double(attrs["scale"]))
        for k in ("batch", "hv", "h", "s_qkv", "s_a", "s_b", "s_st_slot",
                  "s_st_head", "s_dc_slot", "s_dc_head", "s_kc_slot",
                  "s_kc_head", "s_gc_slot", "s_gc_head", "s_su", "s_sm",
                  "s_sf"):
            args.append(ctypes.c_int64(int(attrs[k])))
        args.append(ctypes.c_int64(99))  # mode: full step
        out_t = torch.zeros(out.shape, dtype=out.dtype, device=out.device)
        args.append(ctx.tensor(out_t))
        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        ret = lib.aclnnGdnSketchStepGetWorkspaceSize(
            *args, ctypes.byref(ws_size), ctypes.byref(executor))
        assert ret == 0
        ws = None
        ws_ptr = ctypes.c_void_p()
        if ws_size.value:
            ws = torch.empty(int(ws_size.value), dtype=torch.uint8,
                             device=out.device)
            ws_ptr = ctypes.c_void_p(int(ws.data_ptr()))
        stream = int(torch.npu.current_stream().npu_stream)
        ret = lib.aclnnGdnSketchStep(ws_ptr, ctypes.c_uint64(ws_size.value),
                                     executor, ctypes.c_void_p(stream))
        assert ret == 0
        out.copy_(out_t)
        return True
    except Exception:
        return False
    finally:
        ctx.destroy()
