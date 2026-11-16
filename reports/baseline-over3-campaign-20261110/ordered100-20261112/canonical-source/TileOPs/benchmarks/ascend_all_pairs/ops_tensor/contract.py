"""Metadata-only support decisions, shared by factories and CPU coverage audit."""
from __future__ import annotations

from math import prod

FLOATS = {"float16", "bfloat16", "float32"}
IMPLEMENTED = {
    "AddFwdOp": "add",
    "TruncFwdOp": "trunc",
    "GemmFwdOp": "gemm",
    "BmmFwdOp": "gemm",
    "GroupedGemmFwdOp": "gemm",
    "MoEGroupedGemmFwdOp": "gemm",
    "Conv1dFwdOp": "gemm",
    "Conv2dFwdOp": "gemm",
    "Conv3dFwdOp": "gemm",
    "SSDChunkCouplingFwdOp": "gemm",
    "MaxPool1dFwdOp": "reduction",
    "MaxPool2dFwdOp": "reduction",
    "MaxPool3dFwdOp": "reduction",
    "AdaptiveMaxPool2dFwdOp": "reduction",
    "SumFwdOp": "gemm",
    "MeanFwdOp": "gemm",
    "MeanPoolingFwdOp": "gemm",
    "AvgPool1dFwdOp": "gemm",
    "AvgPool2dFwdOp": "gemm",
    "AvgPool3dFwdOp": "gemm",
    "AdaptiveAvgPool2dFwdOp": "gemm",
}


def tensor_spec(value):
    return (isinstance(value, (tuple, list)) and len(value) == 2
            and isinstance(value[0], (tuple, list)) and isinstance(value[1], str))


def axes(value, rank):
    result = (value,) * rank if isinstance(value, int) else tuple(value)
    if len(result) != rank:
        raise ValueError(f"Expected {rank} spatial coordinates, got {result}")
    return result


def convolution_plan(params):
    shape, dtype = params["input"]
    weight_shape, weight_dtype = params["weight"]
    rank = len(shape) - 2
    stride = axes(params.get("stride", 1), rank)
    padding = axes(params.get("padding", 0), rank)
    dilation = axes(params.get("dilation", 1), rank)
    groups = params.get("groups", 1)
    n, ci, *spatial = shape
    co, cg, *kernel = weight_shape
    if rank not in (1, 2, 3) or len(kernel) != rank or groups < 1:
        raise ValueError("Invalid convolution dimensions/groups")
    if ci != groups * cg or co % groups or dtype != weight_dtype:
        raise ValueError("Convolution channels/dtypes disagree")
    if min(*stride, *dilation, *kernel, n, ci, co) < 1 or min(padding) < 0:
        raise ValueError("Only positive extents/stride/dilation and nonnegative padding are supported")
    output = tuple((s + 2 * p - d * (k - 1) - 1) // t + 1
                   for s, p, d, k, t in zip(spatial, padding, dilation, kernel, stride))
    if min(output) < 1 or tuple(params["output"][0]) != (n, co, *output):
        raise ValueError("Convolution output shape disagrees with manifest")
    return dict(rank=rank, batch=n, groups=groups, channels_per_group=cg,
                outputs_per_group=co // groups, kernel=tuple(kernel), spatial=tuple(spatial),
                stride=stride, padding=padding, dilation=dilation, output=output,
                reduction=cg * prod(kernel), output_positions=prod(output))


def support_reason(op, params):
    """None means implemented for these metadata, NOT numerical verification."""
    if op not in IMPLEMENTED:
        return "No registered complete implementation; see source semantic assessment."
    floats = [v[1] for v in params.values() if tensor_spec(v) and not v[1].startswith(("int", "uint", "bool"))]
    if any(dtype not in FLOATS for dtype in floats):
        return "Original selected templates support FP16/BF16/FP32 only; narrowing is forbidden."
    if op == "AddFwdOp" and params.get("alpha", 1) != 1:
        return "Original Add computes a+b only; scaling alpha would introduce external arithmetic."
    if op.startswith("Conv"):
        try:
            convolution_plan(params)
        except (KeyError, TypeError, ValueError) as exc:
            return str(exc)
    if op == "MoEGroupedGemmFwdOp":
        if params.get("activation") is not None:
            return "Dense original GEMM has no floating gated-activation epilogue; quantized SwiGLU is not equivalent."
        layout = params["layout"]
        if "masked" not in layout:
            spec = layout.get("contiguous", {})
            if spec.get("metadata_kind") != "physical_psum":
                return "This adapter accepts physical prefix sums or masked counts, not per-row routing."
    if op.startswith(("MaxPool", "AvgPool")) or op in ("AdaptiveMaxPool2dFwdOp", "AdaptiveAvgPool2dFwdOp"):
        try:
            pooling_plan(op, params)
        except (KeyError, TypeError, ValueError) as exc:
            return str(exc)
    if op == "MeanPoolingFwdOp" and params.get("accum_dtype", "float32") != "float32":
        return "Mean-pooling contraction route uses the original FP32 GEMM accumulator only."
    return None


def pooling_plan(op, params):
    """Only whole pooling windows fitting the original 64-lane reduction."""
    shape = tuple(params["input"][0])
    is_maximum = "MaxPool" in op
    if op in ("AdaptiveMaxPool2dFwdOp", "AdaptiveAvgPool2dFwdOp"):
        h, w = shape[-2:]
        oh, ow = params["output_size"]
        oh, ow = h if oh is None else oh, w if ow is None else ow
        windows = []
        for i in range(oh):
            hs, he = i * h // oh, ((i + 1) * h + oh - 1) // oh
            for j in range(ow):
                ws, we = j * w // ow, ((j + 1) * w + ow - 1) // ow
                windows.append((i, j, hs, he, ws, we))
        largest = max((he - hs) * (we - ws) for _, _, hs, he, ws, we in windows)
        if is_maximum and largest > 64:
            raise ValueError(f"Adaptive pooling has a {largest}-value window; original reduction handles exactly 64 lanes. A multi-pass reduction is outside scope.")
        return dict(adaptive=True, output=(oh, ow), windows=windows, max_window=largest)
    rank = int(op[len("MaxPool") if is_maximum else len("AvgPool")])
    spatial = shape[-rank:]
    kernel = axes(params["kernel_size"], rank)
    dilation = axes(params.get("dilation", 1), rank)
    stride = axes(params.get("stride") or params["kernel_size"], rank)
    padding = axes(params.get("padding", 0), rank)
    if is_maximum and prod(kernel) > 64:
        raise ValueError(f"Pooling window has {prod(kernel)} values; original single-register reduction handles 64 and cannot be extended by padding.")
    output, right = [], []
    for size, k, d, s, p in zip(spatial, kernel, dilation, stride, padding):
        extent = d * (k - 1) + 1
        out = (size + 2 * p - extent + (s - 1 if params.get("ceil_mode") else 0)) // s + 1
        if params.get("ceil_mode") and (out - 1) * s >= size + p:
            out -= 1
        output.append(out)
        right.append(max(p, (out - 1) * s + extent - size - p))
    if tuple(params["output"][0]) != (*shape[:-rank], *output):
        raise ValueError("Pooling ceil/stride output disagrees with manifest")
    return dict(adaptive=False, rank=rank, output=tuple(output), kernel=kernel, dilation=dilation,
                stride=stride, padding=padding, right=tuple(right), max_window=prod(kernel))
