"""FP8 type specialization of CATLASS's existing per-group/per-block GEMM.

The original epilogue explicitly accepts floating accumulators and FP32 scale
grids. No scales are quantized to MX/FP8, and no input is dequantized by torch.
"""
from __future__ import annotations
import ctypes as C
from pathlib import Path
from .build import build, hashes


class FP8Engine:
    def __init__(self, output_dtype):
        import torch
        from benchmarks.ascend_baselines import _owned_native, UnsupportedCase
        if output_dtype not in (torch.float16, torch.bfloat16):
            raise UnsupportedCase("Original example51 FP8 binding writes FP16 or BF16")
        self.output_dtype = output_dtype
        kind = "fp8_f16" if output_dtype == torch.float16 else "fp8_bf16"
        binary, self.source_sha256, self.receipt = build(kind)
        symbol = "allpairs_catlass_" + kind
        self.lib = _owned_native(binary, [symbol])
        self.fn = getattr(self.lib, symbol)
        self.fn.argtypes = [C.c_void_p] * 6 + [C.c_uint32] * 3 + [C.c_void_p]
        self.fn.restype = C.c_int

    def _launch(self, a, b, sa, sb, bias):
        import torch
        m, k = a.shape
        n = b.shape[0]
        # B input is [N,K], native row-major B is [K,N]. Scales are [K/128,N/128].
        a, b, sa, sb = a.contiguous(), b.T.contiguous(), sa.contiguous(), sb.T.contiguous()
        native_bias = None if bias is None else bias.float().contiguous()
        out = torch.empty((m, n), dtype=self.output_dtype, device=a.device)
        rc = self.fn(a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(),
                     None if native_bias is None else native_bias.data_ptr(), out.data_ptr(), m, n, k,
                     torch.npu.current_stream(a.device).npu_stream)
        if rc:
            raise RuntimeError(f"CATLASS original example51 FP8 launch status {rc}")
        return out

    def __call__(self, a, b, sa, sb, bias=None):
        import torch
        from benchmarks.ascend_baselines import UnsupportedCase
        m, k = a.shape
        n, kb = b.shape
        if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype or kb != k or k % 128:
            raise UnsupportedCase("FP8 binding requires E4M3FN operands and K divisible by128")
        if sa.dtype != torch.float32 or sb.dtype != torch.float32:
            raise UnsupportedCase("Scales must be FP32; no lossy scale conversion")
        blocks = k // 128
        if sa.numel() == 1:
            sa = sa.reshape(1, 1).expand(m, blocks)
        if tuple(sa.shape) != (m, blocks):
            raise UnsupportedCase("A scale must be scalar or [M,K/128]")
        if sb.numel() == 1:
            sb = sb.reshape(1, 1).expand((n + 127) // 128, blocks)
        if tuple(sb.shape) == ((n + 127) // 128, blocks):
            return self._launch(a, b, sa, sb, bias)
        if tuple(sb.shape) == (n, blocks):
            # Original B grid is one scale per128 columns. Independent column
            # calls express arbitrary per-column scales without merging scales,
            # splitting the K reduction, or altering the native schedule.
            out = torch.empty((m, n), dtype=self.output_dtype, device=a.device)
            for column in range(n):
                out[:, column:column + 1].copy_(self._launch(
                    a, b[column:column + 1], sa, sb[column:column + 1],
                    None if bias is None else bias[column:column + 1]))
            return out
        raise UnsupportedCase("B scale must be scalar, [N,K/128], or [ceil(N/128),K/128]")


def _candidate(run, engine, adaptation):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    return Candidate(Implementation(run), "CATLASS example51 QuantMatmulPerGroupPerBlockTla / native FP8 operand + FP32 accumulator type parameters",
                     adaptation, engine.source_sha256 | hashes([Path(__file__)]),
                     timing_mode="host_roundtrip",
                     limitations=("Current ARM64 CPU compilation only; original manifest numeric check still required.",
                                  "Per-column scale route may issue thousands of complete native GEMMs; include every call and copy in timing.",))


def gemm_fp8(case):
    import torch
    dtype = getattr(torch, case.params["out_dtype"])
    engine = FP8Engine(dtype)
    def run(a, b, sa, sb, bias=None):
        return engine(a, b, sa, sb, bias)
    return _candidate(run, engine, "Original FP8/FP32-scale per-block epilogue; broadcast scalar scales, transpose grids each run; per-row B scales use independent output-column calls; optional native FP32 bias input after exact widening.")


def bmm_fp8(case):
    import torch
    engine = FP8Engine(getattr(torch, case.params["out_dtype"]))
    transposed = case.params.get("trans_b", False)
    def run(a, b, sa, sb):
        return torch.stack([engine(a[i], b[i] if transposed else b[i].T, sa, sb)
                            for i in range(a.shape[0])])
    return _candidate(run, engine, "Independent batch slices, native complete scaled FP8 GEMM per batch; current scalar scales broadcast to original FP32 scale grids; input transpose and output stack inside run.")


def moe_grouped_fp8(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    if "masked" not in case.params["layout"]:
        raise UnsupportedCase("This native FP8 adapter covers masked expert slabs")
    engine = FP8Engine(torch.bfloat16)
    def run(a, sa, b, sb, metadata):
        counts = metadata.tolist()
        if len(counts) != a.shape[0] or len(counts) != b.shape[0]:
            raise ValueError("Expert count metadata mismatch")
        out = torch.zeros((*a.shape[:-1], b.shape[1]), dtype=torch.bfloat16, device=a.device)
        for i, count in enumerate(counts):
            if not 0 <= count <= a.shape[1]:
                raise ValueError("Masked expert count outside slab")
            if count:
                out[i, :count].copy_(engine(a[i, :count], b[i], sa[i, :count], sb[i]))
        return out
    return _candidate(run, engine, "Read masked expert counts each run, invoke native full scaled FP8 GEMM for each valid expert slice; preserve per1x128 A/per128x128 B FP32 scales and BF16 output; zero undefined rows only.")
