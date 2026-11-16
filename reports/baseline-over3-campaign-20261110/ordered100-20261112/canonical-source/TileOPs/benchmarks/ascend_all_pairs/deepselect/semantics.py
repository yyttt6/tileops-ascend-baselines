"""Device-free metadata planning shared by factories and the coverage audit.

These are mathematical mappings, not device validation receipts. No torch import.
"""

from math import prod

PROVIDER_OPS = (
    "AmaxFwdOp", "MaximumFwdOp", "ReluFwdOp", "AnyFwdOp", "LogicalOrFwdOp",
    "MaxPool1dFwdOp", "MaxPool2dFwdOp", "MaxPool3dFwdOp", "AdaptiveMaxPool2dFwdOp",
)
HALF_TYPES = ("float16", "bfloat16")
MAX_WIDTH = 2 * 1024 * 1024  # structs.h: strict '<' bound


def axes_plan(shape, dim, keepdim):
    rank = len(shape)
    axes = tuple(range(rank)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    if not axes:
        axes = tuple(range(rank))  # torch reductions treat an empty dim tuple as all axes.
    if rank == 0:
        if dim not in (None, 0, -1, (), []):
            raise ValueError("Invalid scalar reduction axis")
        return (), 1, 1, ()
    if any(not -rank <= a < rank for a in axes):
        raise ValueError("Reduction axis out of range")
    axes = tuple(a % rank for a in axes)
    if len(set(axes)) != len(axes):
        raise ValueError("Duplicate reduction axis")
    kept = tuple(a for a in range(rank) if a not in axes)
    width = prod(shape[a] for a in axes)
    output = tuple(1 if a in axes else shape[a] for a in range(rank)) if keepdim else tuple(shape[a] for a in kept)
    return kept + axes, prod(shape) // width, width, output


def per_axis(value, rank, fallback=None):
    if value is None:
        return tuple(fallback)
    values = (value,) * rank if isinstance(value, int) else tuple(value)
    if len(values) != rank:
        raise ValueError("Pool parameter rank mismatch")
    return tuple(fallback[i] if v is None and fallback is not None else v for i, v in enumerate(values))


def pool_plan(params):
    shape = tuple(params["input"][0])
    spatial = shape[2:]
    rank = len(spatial)
    kernel = per_axis(params["kernel_size"], rank)
    stride = per_axis(params.get("stride"), rank, kernel)
    padding = per_axis(params.get("padding", 0), rank)
    dilation = per_axis(params.get("dilation", 1), rank)
    output, right = [], []
    for n, k, s, p, d in zip(spatial, kernel, stride, padding, dilation):
        effective = d * (k - 1) + 1
        o = (n + 2 * p - effective + (s - 1 if params.get("ceil_mode", False) else 0)) // s + 1
        if params.get("ceil_mode", False) and (o - 1) * s >= n + p:
            o -= 1
        output.append(o)
        right.append(max(p, (o - 1) * s + effective - (n + p)))
    return shape, kernel, stride, padding, dilation, tuple(output), tuple(right)


def adaptive_windows(params):
    shape = tuple(params["input"][0])
    h, w = shape[-2:]
    oh, ow = per_axis(params.get("output_size"), 2, (h, w))
    # Integer floor/ceil boundaries; no input-dependent metadata.
    windows = tuple(
        (i * h // oh, ((i + 1) * h + oh - 1) // oh,
         j * w // ow, ((j + 1) * w + ow - 1) // ow)
        for i in range(oh) for j in range(ow)
    )
    return shape, (oh, ow), windows


def support(op, params):
    """Return (semantic status, reason); never materialize Case.inputs."""
    if op not in PROVIDER_OPS:
        return "unsupported", "No complete legal mapping to the owned Ascend TopK API; see coverage reason."
    key = "x" if op in ("AmaxFwdOp", "AnyFwdOp") else "input"
    shape, dtype = params[key]
    if not shape or min(shape) <= 0:
        return "unsupported", "This adapter requires nonempty, nonscalar manifest tensors."
    boolean = op in ("AnyFwdOp", "LogicalOrFwdOp")
    if boolean and dtype != "bool":
        return "unsupported", "Only bool can be widened losslessly to the 0/1 BF16 selection domain; arbitrary nonzero testing is not a DeepSelect operation."
    if not boolean and dtype not in HALF_TYPES:
        return "unsupported", "Ascend API accepts BF16 lanes only; FP16 has a bit-preserving ordered view, but arbitrary FP32 cannot be represented losslessly. CUDA FP32 is not an Ascend route."
    if op in ("MaximumFwdOp", "LogicalOrFwdOp") and params["other"][1] != dtype:
        return "unsupported", "Mixed input dtypes have no inspected lossless common selection representation."
    if op == "ReluFwdOp" and params.get("inplace", False):
        return "unsupported", "The input-preserving candidate implements only the manifest's out-of-place ReLU."
    if op in ("AmaxFwdOp", "AnyFwdOp"):
        _, _, width, output = axes_plan(shape, params.get("dim"), params.get("keepdim", False))
        if output != tuple(params["output"][0]):
            return "unsupported", "Reduction metadata disagrees with the unchanged manifest output shape."
    elif op.startswith("MaxPool"):
        shape, kernel, _, _, _, out, _ = pool_plan(params)
        width = prod(kernel)
        if tuple(params["output"][0]) != shape[:2] + out:
            return "unsupported", "Pool output geometry disagrees with the unchanged manifest."
    elif op.startswith("Adaptive"):
        shape, out, windows = adaptive_windows(params)
        width = max((he - hs) * (we - ws) for hs, he, ws, we in windows)
        if tuple(params["output"][0]) != shape[:-2] + out:
            return "unsupported", "Adaptive pool output geometry disagrees with the unchanged manifest."
    else:
        width = 2
    if not 0 < width < MAX_WIDTH:
        return "unsupported", f"Packed selection width {width} violates the original 0 < width < {MAX_WIDTH} ABI bound."
    return "supported_unverified", "Complete value output is expressible by original DeepSelect TopK k=1 with layout/padding or exact 0/1 dtype adaptation; device checks are prohibited in this task."
