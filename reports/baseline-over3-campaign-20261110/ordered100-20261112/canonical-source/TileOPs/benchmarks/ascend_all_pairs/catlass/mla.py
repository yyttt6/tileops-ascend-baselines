"""Layout adapter for the original example19 complete MLA decode host API.

The AtlasA2 entry is separately compatibility-compiled, without changing its
architecture tags or schedule. It is not registered while that build is blocked.
"""
from __future__ import annotations
import ctypes as C
from pathlib import Path
from .build import hashes
from .legacy_build import build


def mla_decode(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, _owned_native
    q, qr, k, kr = case.inputs
    if (q.dtype not in (torch.float16, torch.bfloat16) or any(t.dtype != q.dtype for t in (qr, k, kr))
            or q.shape[-1] != 512 or qr.shape[-1] != 64 or k.shape[-2:] != (1, 512)
            or kr.shape[-2:] != (1, 64) or q.shape[1] not in (64, 128) or k.shape[1] % 128):
        raise UnsupportedCase("Original MLA decode bridge requires FP16/BF16, latent512+rope64, one KVhead,64/128 Qheads and128-aligned cache length")
    binary, shas, _ = build()
    lib = _owned_native(binary, ["allpairs_catlass_mla"])
    fn = lib.allpairs_catlass_mla
    fn.argtypes = [C.c_void_p] * 6 + [C.c_uint32] * 3 + [C.c_int, C.c_void_p]
    fn.restype = C.c_int
    def run(q, qr, k, kr):
        b, h, _ = q.shape
        length = k.shape[1]
        # Existing dense KV rows are a contiguous128-token physical page pool.
        # The table is generated each call, never persisted input-dependent data.
        page_table = torch.arange(b * (length // 128), dtype=torch.int32, device=q.device).reshape(b, -1)
        q, qr, k, kr = (t.contiguous() for t in (q, qr, k, kr))
        output = torch.empty_like(q)
        rc = fn(q.data_ptr(), qr.data_ptr(), k.data_ptr(), kr.data_ptr(), page_table.data_ptr(), output.data_ptr(),
                b, h, length, 0 if q.dtype == torch.float16 else 1, torch.npu.current_stream(q.device).npu_stream)
        if rc:
            raise RuntimeError(f"Original CATLASS MLA host status {rc}")
        return output
    return Candidate(Implementation(run), "CATLASS example19 CatlassKernel::Mla / original MLA+TP strategy selection",
                     "Per-call contiguous operands and logical-to-physical128-token page table; original host builds lengths/tiling/workspaces and dispatches its unchanged full512+64 attention; complete output and protected inputs.",
                     shas | hashes([Path(__file__)]), timing_mode="host_roundtrip",
                     limitations=("AtlasA2 source on3510 requires a successful compatibility build and separately authorized numerical/device ownership checks; no current-host device run.",))
