"""Neutral-argument adapters to ATVC's original Addcmul kernel."""


def addcmul(case):
    import ctypes as C
    from math import prod
    from pathlib import Path

    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import (
        Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _owned_native, source_hashes,
    )

    op = case._call.signature.name
    x, *others = case.inputs
    boolean = op in ("LogicalAndFwdOp", "LogicalNotFwdOp")
    if boolean and any(t.dtype != torch.bool for t in case.inputs):
        raise UnsupportedCase("Only boolean 0/1 inputs have a neutral Addcmul logical mapping")
    if not boolean and any(t.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                           for t in case.inputs):
        raise UnsupportedCase("ATVC Addcmul binds its original FP32 example; this integer conversion is not lossless")
    shape = torch.broadcast_shapes(x.shape, others[0].shape) if others else x.shape
    output_dtype = getattr(torch, case.params["output"][1])
    count = prod(shape)
    if count <= 0:
        raise UnsupportedCase("ATVC's broadcast tiler requires positive extents")
    if op not in ("MulFwdOp", "SubFwdOp", "NegFwdOp", "LogicalAndFwdOp", "LogicalNotFwdOp"):
        raise UnsupportedCase("No neutral Addcmul parameter mapping for this operator")
    binary = WORKSPACE / ".cache/all-pairs/atvc/libatvc_addcmul.so"
    lib = _owned_native(binary, ["atvc_addcmul_plan", "atvc_addcmul_run", "atvc_addcmul_destroy"])
    lib.atvc_addcmul_plan.argtypes = [C.c_int64, C.POINTER(C.c_uint64)]
    lib.atvc_addcmul_plan.restype = C.c_void_p
    lib.atvc_addcmul_run.argtypes = [C.c_void_p] * 5 + [C.c_float] + [C.c_void_p] * 2
    lib.atvc_addcmul_run.restype = C.c_int
    lib.atvc_addcmul_destroy.argtypes = [C.c_void_p]
    lib.atvc_addcmul_destroy.restype = None
    workspace_bytes = C.c_uint64()
    plan = lib.atvc_addcmul_plan(count, C.byref(workspace_bytes))
    if not plan:
        raise UnsupportedCase("ATVC CalcBroadcastTiling rejected the duplicate-row layout")
    workspace = torch.empty(max(1, workspace_bytes.value), dtype=torch.uint8, device=x.device)
    output = torch.empty((2, count), dtype=torch.float32, device=x.device)
    # Input-independent neutral constants only; negative zero preserves product zero signs.
    base = torch.full_like(output, 1.0 if op == "LogicalNotFwdOp" else -0.0)
    ones = torch.ones(count, dtype=torch.float32, device=x.device)
    coefficient = -float(case.params.get("alpha", 1)) if op == "SubFwdOp" else (-1.0 if op in ("NegFwdOp", "LogicalNotFwdOp") else 1.0)

    def pack(value):
        return value.to(torch.float32).expand(shape).contiguous().reshape(-1)

    def run(value, other=None):
        left = pack(value)
        if op in ("MulFwdOp", "LogicalAndFwdOp"):
            first, second = left, pack(other)
        elif op == "SubFwdOp":
            first, second = pack(other), ones
            base.copy_(left.reshape(1, -1).expand_as(base))
        else:
            first, second = left, ones
        rc = lib.atvc_addcmul_run(plan, first.data_ptr(), second.data_ptr(), base.data_ptr(),
                                  output.data_ptr(), coefficient, workspace.data_ptr(),
                                  torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f"ATVC original Addcmul launch returned {rc}")
        return output[0].reshape(shape).to(output_dtype)

    vendor = LIBRARIES / "atvc"
    source = Path(__file__).parent
    return Candidate(
        Implementation(run), "ATVC::AddcmulCustom<original FP32 Traits, original BroadcastPolicy>",
        "Neutral arguments implement multiplication/subtraction/negation and bool AND/NOT; exact bool/FP16/BF16 widening. Per-call expand/contiguous, duplicate-row broadcast and crop; original kernel and both policies unchanged.",
        source_hashes(Path(__file__), source / "addcmul_bridge.cpp", binary,
                      *sorted((vendor / "examples/addcmul").glob("*.h")),
                      *sorted((vendor / "include").rglob("*.h"))),
        cleanup=lambda: lib.atvc_addcmul_destroy(plan),
    )
