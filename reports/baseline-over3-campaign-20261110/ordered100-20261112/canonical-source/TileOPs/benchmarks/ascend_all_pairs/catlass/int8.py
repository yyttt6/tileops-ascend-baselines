"""Original INT8 full-dequant entry, including independent token/channel scales."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
from .build import build, hashes


def int8_gemm(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, _owned_native
    if case.params["out_dtype"] != "bfloat16" or "bias" in case.params:
        raise UnsupportedCase("This original example57 instantiation is BF16 output, no bias")
    binary, source_sha256, _ = build("int8")
    lib = _owned_native(binary, ["allpairs_catlass_int8"])
    fn = lib.allpairs_catlass_int8
    fn.argtypes = [C.c_void_p] * 5 + [C.c_uint32] * 3 + [C.c_void_p]
    fn.restype = C.c_int
    def run(a, b, scale_a, scale_b, bias=None):
        if bias is not None or a.dtype != torch.int8 or b.dtype != torch.int8:
            raise UnsupportedCase("Original INT8 entry requires INT8 operands and no bias")
        m, k = a.shape
        n, kb = b.shape
        if kb != k or scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
            raise ValueError("Invalid INT8 reduction or scale dtype")
        # Scalar expansion into the existing token/channel API is lossless.
        sa = scale_a.reshape(-1).expand(m).contiguous()
        sb = scale_b.reshape(-1).expand(n).contiguous()
        a, b = a.contiguous(), b.T.contiguous()
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        rc = fn(a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(), m, n, k,
                torch.npu.current_stream(a.device).npu_stream)
        if rc:
            raise RuntimeError(f"CATLASS original INT8 full-dequant status {rc}")
        return out
    return Candidate(Implementation(run), "CATLASS example57 KernelMatmulDequant per-token/per-channel BF16",
                     "Per-call B transpose and exact scale broadcast; native INT32 accumulation, original FP32 scaling and BF16 output. No torch arithmetic.",
                     source_sha256 | hashes([Path(__file__)]),
                     timing_mode="host_roundtrip",
                     limitations=("CPU compile/static checks only; no current-host numerical receipt.",))
