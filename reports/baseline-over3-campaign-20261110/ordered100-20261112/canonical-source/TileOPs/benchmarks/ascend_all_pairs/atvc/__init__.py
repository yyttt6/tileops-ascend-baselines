"""ATVC's original Add, ReduceSum and Addcmul templates with manifest adapters."""

from .addcmul import addcmul


def add(case):
    from benchmarks.ascend_extra_baselines import atvc_add

    return atvc_add(case)


def reduce_sum(case):
    import ctypes as C
    from math import prod
    from pathlib import Path

    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import (
        Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _owned_native, source_hashes,
    )

    x, = case.inputs
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase("Original ATVC ReduceSum floating template requires exact FP32 input representation")
    if not x.ndim or not prod(x.shape):
        raise UnsupportedCase("Original ATVC ReduceSum tiler requires nonempty positive-rank input")
    dim = case.params.get("dim")
    axes = list(range(x.ndim)) if dim is None else ([dim] if isinstance(dim, int) else list(dim))
    if not axes:
        axes = list(range(x.ndim))
    axes = [axis % x.ndim for axis in axes]
    if len(set(axes)) != len(axes):
        raise UnsupportedCase("Duplicate reduction axes")
    out_shape = tuple(1 if i in axes else size for i, size in enumerate(x.shape))
    if not case.params.get("keepdim", False):
        out_shape = tuple(size for i, size in enumerate(x.shape) if i not in axes)
    dtype = case.params.get("dtype")
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else (dtype or x.dtype)
    source = Path(__file__).with_name("reduce_bridge.cpp")
    binary = WORKSPACE / ".cache/all-pairs/atvc/libatvc_sum.so"
    lib = _owned_native(binary, ["atvc_sum_plan", "atvc_sum_run", "atvc_sum_destroy"])
    lib.atvc_sum_plan.argtypes = [C.POINTER(C.c_int64), C.c_int, C.POINTER(C.c_int64), C.c_int,
                                 C.POINTER(C.c_uint64)]
    lib.atvc_sum_plan.restype = C.c_void_p
    lib.atvc_sum_run.argtypes = [C.c_void_p] * 5
    lib.atvc_sum_run.restype = C.c_int
    lib.atvc_sum_destroy.argtypes = [C.c_void_p]
    lib.atvc_sum_destroy.restype = None
    workspace_bytes = C.c_uint64()
    plan = lib.atvc_sum_plan((C.c_int64 * x.ndim)(*x.shape), x.ndim,
                             (C.c_int64 * len(axes))(*axes), len(axes), C.byref(workspace_bytes))
    if not plan:
        raise UnsupportedCase("ATVC original CalcReduceTiling rejected this shape and axis combination")
    workspace = torch.empty(max(1, workspace_bytes.value), dtype=torch.uint8, device=x.device)
    output = torch.empty(out_shape, dtype=torch.float32, device=x.device)

    def run(value):
        packed = value.to(torch.float32).contiguous()
        rc = lib.atvc_sum_run(plan, packed.data_ptr(), output.data_ptr(), workspace.data_ptr(),
                              torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f"ATVC original ReduceSum launch returned {rc}")
        return output.to(dtype)

    root = LIBRARIES / "atvc"
    return Candidate(
        Implementation(run), "ATVC::ReduceSumCustom<ReduceOpTraitsFloat, original_policy>",
        "Exact 16-bit to FP32 widening, contiguous layout, original tiler and all 23 original policies; cast output to manifest dtype.",
        source_hashes(Path(__file__), source, binary,
                      root / "examples/ops_pytorch/reduce_sum/reduce_sum_impl.h",
                      *sorted((root / "include").rglob("*.h"))),
        cleanup=lambda: lib.atvc_sum_destroy(plan),
    )


PROVIDERS = {"AddFwdOp": add, "SumFwdOp": reduce_sum}
PROVIDERS.update(dict.fromkeys(
    ("MulFwdOp", "SubFwdOp", "NegFwdOp", "LogicalAndFwdOp", "LogicalNotFwdOp"), addcmul
))
