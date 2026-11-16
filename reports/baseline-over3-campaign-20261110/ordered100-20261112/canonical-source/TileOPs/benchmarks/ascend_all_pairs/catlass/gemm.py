"""Layout-only adapters to CATLASS's original example43 matmul templates."""
from __future__ import annotations

import ctypes as C
from math import prod
from pathlib import Path

from .build import build, hashes


class GemmEngine:
    """No cached operand data. Each call packs the operands supplied to that call."""

    def __init__(self, dtype, *, bias=False, output_dtype=None):
        import torch
        from benchmarks.ascend_baselines import UnsupportedCase, _owned_native
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise UnsupportedCase("CATLASS example43 operand types: FP16/BF16/FP32.")
        self.dtype = dtype
        self.output_dtype = output_dtype or dtype
        if self.output_dtype not in (dtype, torch.float32):
            raise UnsupportedCase("No lossy input conversions or post-matmul emulation.")
        self.bias = bias
        suffix = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}[dtype]
        kind = {"f16": "gemm_f16", "bf16": "gemm", "f32": "gemm_f32"}[suffix]
        symbol = f"classic_catlass_gemm_{suffix}"
        if self.output_dtype == torch.float32 and dtype != torch.float32:
            kind = f"gemm_{suffix}_f32"
            symbol = f"classic_catlass_gemm_{suffix}_f32"
        if bias:
            if dtype == torch.float32:
                raise UnsupportedCase("Original FP32 bias tile exceeds L1; do not change the schedule.")
            if self.output_dtype != dtype:
                raise UnsupportedCase("The native bias bridge preserves the operand output dtype.")
            kind, symbol = "gemm_bias", "classic_catlass_gemm_bias"
        binary, self.source_sha256, self.receipt = build(kind)
        self.lib = _owned_native(binary, [symbol])
        self.fn = getattr(self.lib, symbol)
        self.fn.argtypes = ([C.c_void_p] * 4 + [C.c_uint32] * 3 + [C.c_int, C.c_void_p]
                            if bias else [C.c_void_p] * 3 + [C.c_uint32] * 3 + [C.c_void_p])
        self.fn.restype = C.c_int

    def __call__(self, a, b, bias=None):
        import torch
        if a.dtype != self.dtype or b.dtype != self.dtype or a.device != b.device:
            raise ValueError("Operands changed dtype/device after candidate construction")
        m, k = a.shape
        kb, n = b.shape
        if k != kb or min(m, n, k) <= 0:
            raise ValueError("GEMM expects positive M,N,K and matching reduction dimensions")
        if self.bias:
            if bias is None or bias.dtype != self.dtype or tuple(bias.shape) != (n,):
                raise ValueError("Native bias must be one same-dtype value per output column")
            a, b, bias = a.contiguous(), b.contiguous(), bias.float().contiguous()
            out = torch.empty((m, n), dtype=self.output_dtype, device=a.device)
            rc = self.fn(a.data_ptr(), b.data_ptr(), bias.data_ptr(), out.data_ptr(), m, n, k,
                         {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[self.dtype],
                         torch.npu.current_stream(a.device).npu_stream)
            if rc:
                raise RuntimeError(f"CATLASS example43 bias status {rc}")
            return out
        # The optest instantiation selects the original M<=N swizzle.
        transposed = m > n
        lhs, rhs = (b.T, a.T) if transposed else (a, b)
        lhs, rhs = lhs.contiguous(), rhs.contiguous()
        native_m, native_n = (n, m) if transposed else (m, n)
        out = torch.empty((native_m, native_n), dtype=self.output_dtype, device=a.device)
        rc = self.fn(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), native_m, native_n, k,
                     torch.npu.current_stream(a.device).npu_stream)
        if rc:
            raise RuntimeError(f"CATLASS example43 status {rc}")
        return out.T.contiguous() if transposed else out


def _candidate(run, engine, adaptation):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    return Candidate(Implementation(run), "CATLASS example43 BasicMatmulTla / original optest type instantiation",
                     adaptation, engine.source_sha256 | hashes([Path(__file__)]),
                     timing_mode="host_roundtrip",
                     limitations=("CPU compile/static checks only; no current-host numerical receipt.",))


def gemm(case):
    a, b = case.inputs
    engine = GemmEngine(a.dtype)
    ta, tb = case.params.get("trans_a", False), case.params.get("trans_b", False)
    def run(a, b):
        return engine(a.T if ta else a, b.T if tb else b)
    return _candidate(run, engine, "Per-call transpose/contiguous views; original FP16/BF16/FP32 template and scheduler.")


def bmm(case):
    a, b = case.inputs
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
        raise ValueError("BMM expects equal rank-3 batch dimensions")
    engine = GemmEngine(a.dtype)
    def run(a, b):
        import torch
        return torch.stack([engine(a[i], b[i]) for i in range(a.shape[0])])
    return _candidate(run, engine, "Independent batch slices; native complete GEMM per batch; stack outputs inside run.")


def grouped_gemm(case):
    a, b, *_ = case.inputs
    engine = GemmEngine(a.dtype)
    ta, tb = case.params["transpose_a"], case.params["transpose_b"]
    def run(a, b, batch_sizes, batch_offsets):
        import torch
        # Metadata is an input, not a specialization constant. Read afresh each call.
        sizes, offsets = batch_sizes.tolist(), batch_offsets.tolist()
        running = 0
        for size, offset in zip(sizes, offsets, strict=True):
            if size < 0 or offset != running:
                raise ValueError("GroupedGemm requires nonnegative contiguous size/offset metadata")
            running += size
        if running != a.shape[0]:
            raise ValueError("GroupedGemm sizes must span A")
        outputs = []
        for i, (size, start) in enumerate(zip(sizes, offsets, strict=True)):
            end = start + size
            if ta:
                lhs = a[start:end].T
                rhs = b[:, start:end].T if tb else b[start:end]
            else:
                lhs = a[start:end]
                rhs = b[i].T if tb else b[i]
            if size == 0:
                # Empty group: empty output rows, or the empty-reduction identity.
                outputs.append(torch.zeros((lhs.shape[0], rhs.shape[1]), dtype=a.dtype, device=a.device))
            else:
                outputs.append(engine(lhs, rhs))
        return torch.stack(outputs) if ta else torch.cat(outputs, dim=0)
    return _candidate(run, engine, "Read current group metadata each run; slice whole independent groups, preserve NN/NT/TN/TT contracts; native complete reduction per group.")


def moe_grouped_gemm(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    a, b, _ = case.inputs
    p = case.params
    if p.get("activation") is not None:
        raise UnsupportedCase("Plain original GEMM does not implement the requested fused gated activation.")
    layout = p["layout"]
    if "contiguous" in layout:
        spec = layout["contiguous"]
        if spec.get("packing") != "tight" or spec.get("metadata_kind") != "physical_psum":
            raise UnsupportedCase("Only current physical cumulative offsets / tight contiguous groups are adapted.")
    elif "masked" not in layout:
        raise UnsupportedCase("Unknown expert layout")
    out_dtype = getattr(torch, p["out_dtype"]) if p.get("out_dtype") else a.dtype
    engine = GemmEngine(a.dtype, output_dtype=out_dtype)
    def run(a, b, metadata):
        values = metadata.tolist()
        if len(values) != b.shape[0]:
            raise ValueError("One metadata entry per expert is required")
        n = b.shape[1]
        shape = (*a.shape[:-1], n)
        out = torch.zeros(shape, device=a.device, dtype=out_dtype)
        start = 0
        for expert, value in enumerate(values):
            if "masked" in layout:
                if not 0 <= value <= a.shape[1]:
                    raise ValueError("Masked expert count out of range")
                if value:
                    out[expert, :value].copy_(engine(a[expert, :value], b[expert].T))
            else:
                if not start <= value <= a.shape[0]:
                    raise ValueError("Expert offsets must be monotonic within A")
                if value > start:
                    out[start:value].copy_(engine(a[start:value], b[expert].T))
                start = value
        if "contiguous" in layout and start != a.shape[0]:
            raise ValueError("Expert offsets do not span A")
        return out
    return _candidate(run, engine, "Read expert offsets/counts each call; complete native GEMM per independent expert, FP32 output uses original native output type; zero only undefined/empty rows.")
