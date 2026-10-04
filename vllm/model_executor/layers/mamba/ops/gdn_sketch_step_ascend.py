# SPDX-License-Identifier: Apache-2.0
"""AscendC GDN SketchSSM decode step (A3) — ctypes binding wrapper.

Requires the custom opp package (gdn_sketch_step) to be installed and its
set_env sourced; falls back cleanly when the op is unavailable so the
SKETCHSSM_ASCENDC switch can stay on across environments.
"""

import ctypes
import glob
import os

import torch

_B = None
_AVAILABLE = None
_CACHE = {}
_CACHE_CAP = 0  # executor caching pins input storages (multi-GB state); disabled
_BOUNCE = {}



def _bounce(device, key, like, copy=True, dtype=None):
    """Copy a per-step tensor into a persistent buffer so the aclnn
    executor cache key stays stable; the copy is an async device op.
    ``dtype`` pins the op-def contract (e.g. int64 write_pos)."""
    if dtype is not None and like.dtype != dtype:
        like = like.to(dtype)
    buf = _BOUNCE.get((device, key))
    if (buf is None or buf.shape != like.shape or buf.dtype != like.dtype):
        buf = torch.empty(like.shape, dtype=like.dtype, device=device)
        _BOUNCE[(device, key)] = buf
        if not copy:
            return buf
    buf.copy_(like)
    return buf


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
    custom op is unavailable (caller should fall back to triton).

    The aclnn executor/workspace are cached per (tensor addresses, out
    address, scale): building them costs ~2 ms of host-side aclTensor
    marshalling per call, which dominates small-batch decode.  In graph
    capture mode the addresses are static, so steady-state hits are the
    norm; eager mode churns addresses, misses, and stays correct (the
    cache is capped and cleared to bound growth).
    """
    # the raw aclnn launch is not graph-capture safe; let the caller fall
    # back to the triton path while a capture stream is recording
    if os.environ.get("SKETCHSSM_ASCENDC_DISABLED", "0") == "1":
        return False  # A/B: launcher flow identical, kernel skipped
    try:
        if torch.npu.is_current_stream_capturing():
            return False
    except Exception:
        return False
    bnd = _binding()
    if bnd is None:
        return False
    rt, runtime, lib = bnd
    s = sketch
    if slots.dim() == 2:
        slots = slots[:, 0]
    dev = out.device
    # per-step tensors ride persistent bounce buffers so the executor key
    # (which bakes addresses) is stable across decode steps; the sketch
    # state tensors are long-lived cache allocations and pass through
    mixed_qkv = _bounce(dev, "qkv", mixed_qkv, dtype=torch.bfloat16)
    a = _bounce(dev, "a", a, dtype=torch.float32)
    b = _bounce(dev, "b", b, dtype=torch.float32)
    A_log = _bounce(dev, "alog", A_log, dtype=torch.float32)
    dt_bias = _bounce(dev, "dt", dt_bias, dtype=torch.float32)
    slots = _bounce(dev, "slots", slots, dtype=torch.int32)
    write_pos = _bounce(dev, "wp", write_pos, dtype=torch.int64)
    meta = _bounce(dev, "meta", meta, dtype=torch.int32)
    bout = _bounce(dev, "out", out, copy=False, dtype=torch.bfloat16)
    tensors = (mixed_qkv, a, b, A_log, dt_bias, state, d_cache, k_cache,
               g_cache, slots, write_pos, meta, s.u, s.phi, s.fs, s.beta,
               s.current_d, s.current_k, s.tables.ranks, s.tables.layout)
    stream = int(torch.npu.current_stream().npu_stream)
    key = (tuple(int(t.data_ptr()) for t in tensors), int(bout.data_ptr()),
           float(scale))
    hit = _CACHE.get(key)
    if hit is not None:
        executor, ws = hit
        ret = lib.aclnnGdnSketchStep(
            ctypes.c_void_p(int(ws.data_ptr())), ctypes.c_uint64(ws.numel()),
            executor, ctypes.c_void_p(stream))
        if ret == 0 and bout.data_ptr() != out.data_ptr():
            out.copy_(bout)
        return ret == 0
    attrs = dict(
        scale=scale, batch=mixed_qkv.shape[0], hv=state.shape[1],
        h=k_cache.shape[1], s_qkv=mixed_qkv.stride(0), s_a=a.stride(0),
        s_b=b.stride(0), s_st_slot=state.stride(0), s_st_head=state.stride(1),
        s_dc_slot=d_cache.stride(0), s_dc_head=d_cache.stride(1),
        s_kc_slot=k_cache.stride(0), s_kc_head=k_cache.stride(1),
        s_gc_slot=g_cache.stride(0), s_gc_head=g_cache.stride(1),
        s_su=s.u.stride(0), s_sm=s.phi.stride(0), s_sf=s.fs.stride(0))
    import sys
    if not hasattr(_bounce, "_dbg"):
        _bounce._dbg = 0
    _bounce._dbg += 1
    if _bounce._dbg % 50 == 1 or _bounce._dbg < 4:
        print(f"[gdn-asc] call#{_bounce._dbg} batch={mixed_qkv.shape[0]} "
              f"slots[{int(slots.min())},{int(slots.max())}] "
              f"wp[{int(write_pos.min())},{int(write_pos.max())}] "
              f"meta[{int(meta.min())},{int(meta.max())}] "
              f"qkv={tuple(mixed_qkv.shape)} out={tuple(out.shape)} "
              f"u={tuple(s.u.shape)}", file=sys.stderr, flush=True)
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
        args.append(ctx.tensor(bout))
        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        ret = lib.aclnnGdnSketchStepGetWorkspaceSize(
            *args, ctypes.byref(ws_size), ctypes.byref(executor))
        if ret != 0:
            import sys
            print(f"[gdn-asc] get_ws ret={ret} ws={ws_size.value}",
                  file=sys.stderr, flush=True)
        assert ret == 0, f"get_ws ret={ret} ws={ws_size.value}"
        ws = torch.empty(int(ws_size.value) if ws_size.value else 1,
                         dtype=torch.uint8, device=out.device)
        ret = lib.aclnnGdnSketchStep(
            ctypes.c_void_p(int(ws.data_ptr())),
            ctypes.c_uint64(ws_size.value), executor,
            ctypes.c_void_p(stream))
        if ret != 0:
            import sys
            print(f"[gdn-asc] launch ret={ret}", file=sys.stderr, flush=True)
        assert ret == 0, f"launch ret={ret}"
        if _CACHE_CAP > 0:
            if len(_CACHE) >= _CACHE_CAP:
                _CACHE.clear()
            _CACHE[key] = (executor, ws)
        if bout.data_ptr() != out.data_ptr():
            out.copy_(bout)
        return True
    except Exception as e:
        import sys
        print(f"[gdn-asc] slow-path exception: {type(e).__name__}: {e!r}",
              file=sys.stderr, flush=True)
        return False
    finally:
        ctx.destroy()
