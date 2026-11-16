"""CPU-only semantic decisions shared by factories and coverage generation.

No tensor/framework imports. This is a decision about the inspected implementation,
not a device verification result or a proof that no other implementation can exist.
"""
from math import prod
from itertools import product
import ast

ATTENTION = {
    "GQADenseFwdOp", "GQAVarlenFwdOp", "GQAPagedFwdOp",
    "GQAPrefillPagedWithKVCacheFwdOp", "DSADecodeWithKVCacheFwdOp",
    "NSAVarlenFwdOp", "NSACompressedVarlenFwdOp",
    "MLAVarlenFwdOp",
}
REDUCTION = {"AmaxFwdOp", "LogSumExpFwdOp", "MeanFwdOp", "AnyFwdOp"}
POOL = {"MaxPool1dFwdOp", "MaxPool2dFwdOp", "MaxPool3dFwdOp",
        "AdaptiveMaxPool2dFwdOp", "AvgPool1dFwdOp", "AdaptiveAvgPool2dFwdOp"}
ROUTES = ATTENTION | REDUCTION | POOL | {"MeanPoolingFwdOp", "BmmFwdOp", "MaximumFwdOp", "ReluFwdOp", "LogicalOrFwdOp"}


def mutated_inputs(signature, params):
    """Resolve manifest boolean mutation predicates without importing its runtime."""
    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            return bool(params.get(node.id, False))
        if isinstance(node, ast.BoolOp):
            values = [evaluate(value) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not evaluate(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "present" and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
            return node.args[0].id in params
        raise ValueError("Unsupported manifest mutation predicate")
    result = []
    for name, value in signature.get("inputs", {}).items():
        flag = value.get("mutated", False)
        active = evaluate(ast.parse(flag, mode="eval").body) if isinstance(flag, str) else flag
        if active and name in params:
            result.append(name)
    return result


def axes_plan(shape, dim, keepdim=False):
    rank = len(shape)
    dims = tuple(range(rank)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    if not dims:
        dims = tuple(range(rank))
    if rank == 0:
        return (), (), 1, ()
    if any(d < -rank or d >= rank for d in dims):
        raise ValueError("Reduction axis out of bounds")
    axes = tuple(d % rank for d in dims)
    if len(set(axes)) != len(axes):
        raise ValueError("Repeated reduction axis")
    remaining = tuple(d for d in range(rank) if d not in axes)
    output = tuple(1 if d in axes else n for d, n in enumerate(shape)) if keepdim else tuple(shape[d] for d in remaining)
    return remaining, axes, prod(shape[d] for d in axes), output


def per_axis(value, ndim, default=None):
    if value is None:
        value = default
    return (value,) * ndim if isinstance(value, int) else tuple(value)


def pool_window(op, p, spatial, outspatial, coord):
    """Pure integer layout planner, also checked without any tensor framework."""
    ndim = len(spatial)
    adaptive, average = op.startswith("Adaptive"), "Avg" in op
    if adaptive:
        starts = [c * n // o for c, n, o in zip(coord, spatial, outspatial)]
        stops = [((c + 1) * n + o - 1) // o for c, n, o in zip(coord, spatial, outspatial)]
        dimensions = [range(a, b) for a, b in zip(starts, stops)]
    else:
        kernel = per_axis(p["kernel_size"], ndim)
        stride = per_axis(p.get("stride"), ndim, kernel)
        padding = per_axis(p.get("padding", 0), ndim)
        dilation = per_axis(p.get("dilation", 1), ndim)
        starts = [c * s - pad for c, s, pad in zip(coord, stride, padding)]
        dimensions = [range(a, a + k * d, d) for a, k, d in zip(starts, kernel, dilation)]
    locations, included, source_valid = [], [], []
    for point in product(*dimensions):
        real = all(0 <= c < n for c, n in zip(point, spatial))
        inc = real or (average and p.get("count_include_pad", True) and not adaptive
                       and all(-pad <= c < n + pad for c, n, pad in zip(point, spatial, padding)))
        index = 0
        for c, n in zip(point, spatial):
            index = index * n + min(max(c, 0), n - 1)
        locations.append(index)
        included.append(inc)
        source_valid.append(real)
    return locations, included, source_valid


def classify(op, p):
    """Return (legal_route, precise_reason); never instantiate a workload."""
    if op == "LogicalOrFwdOp":
        if p["input"][1] != "bool" or p["other"][1] != "bool":
            return False, "Only bool0/1 widens losslessly to the native BF16 maximum domain; arbitrary truth predicates are not native."
        return True, "Broadcast/stack complete bool operands, widen exact0/1 to BF16, select original sparse-prefill max statistic and cast0/1 back to bool."
    if op == "AnyFwdOp":
        if p["x"][1] != "bool":
            return False, "Only boolean Any maps losslessly to max(0/1); arbitrary signed/complex inputs require a nonzero predicate absent from the native API."
        return True, "Boolean0/1 converts exactly to native BF16; exposed max maps exactly back to boolean Any with no numerical preprocessing."
    if op in {"MaximumFwdOp", "ReluFwdOp"}:
        if p.get("inplace", False):
            return False, "The selected route preserves inputs and does not implement an in-place mutation contract."
        names = ("input", "other") if op == "MaximumFwdOp" else ("input",)
        if any(p[name][1] != "bfloat16" for name in names):
            return False, "Native max-statistic route requires BF16 values; narrowing this input dtype would lose information."
        return True, "Pack the two broadcast operands (or input and zero for ReLU) as a width2 reduction; select exposed native maximum via the existing basis-query layout."
    if op == "BmmFwdOp":
        if p["a"][1] != "bfloat16" or p["b"][1] != "bfloat16":
            return False, "Native QK operands are BF16 only; FP16 cannot be narrowed losslessly."
        if p["a"][0][-1] > 512:
            return False, "BMM contraction width exceeds native D512; splitting K would require unsupported partial-sum accumulation or lossy intermediate narrowing."
        return True, "Map each B column to a single selected key, A rows to query heads; native max_logits over one key equals the unchanged full QK dot product. Transpose/crop the exposed statistic."
    if op in ATTENTION:
        if op == "DSADecodeWithKVCacheFwdOp":
            names = ("q", "kv")
        elif op == "NSACompressedVarlenFwdOp":
            names = ("q", "k_cmp", "v_cmp")
        elif op == "GQAPagedFwdOp":
            names = ("q", "k_pages", "v_pages")
        elif op == "GQAPrefillPagedWithKVCacheFwdOp":
            names = ("q", "k_new", "v_new", "k_pages", "v_pages")
        elif op == "MLAVarlenFwdOp":
            names = ("q", "k_nope", "v", "k_pe")
        else:
            names = ("q", "k", "v")
        if any(p[n][1] != "bfloat16" for n in names):
            return False, "Native sparse prefill accepts BF16 only; arbitrary FP16/FP32 and scaled FP8 cannot be losslessly converted to BF16."
        if p.get("pos_encoding_mode", "none") != "none" or p.get("fuse_rope", False):
            return False, "RoPE requires arithmetic absent from the Ascend sparse API; fused norm/RoPE APIs are CUDA-only."
        if p.get("softcap") is not None and p["softcap"] > 0:
            return False, "Native sparse scores have scale only, without the requested tanh softcap."
        if p.get("out_dtype") not in (None, "bfloat16"):
            return False, "Native attention output is rounded to BF16; widening it does not implement a full-precision output contract."
        if op != "GQAPrefillPagedWithKVCacheFwdOp" and any(n in p for n in ("q_scale", "k_scale", "v_scale", "rope_cos", "rope_sin")):
            return False, "Scaled or rotary input slots require arithmetic beyond layout/dtype adaptation."
        qdim = p["q"][0][-1]
        width = qdim if op == "DSADecodeWithKVCacheFwdOp" else qdim + p[names[2]][0][-1]
        if width > 512:
            return False, f"Required score/value embedding width {width} exceeds native fixed width512; cropping changes scores or values."
        qheads, kheads = p["q"][0][-2], p[names[1]][0][-2]
        if qheads % kheads:
            return False, "Query heads are not divisible by KV heads."
        if op == "DSADecodeWithKVCacheFwdOp" and not p.get("is_causal", True):
            return False, "Unresolved oracle mismatch: DSADecodeWorkload.selection_mask always applies causal compression even when is_causal=False."
        return True, "BF16 native sparse attention via per-call layout packing, head padding and metadata-to-index conversion; native scale/sinks preserved."
    if op in REDUCTION | POOL | {"MeanPoolingFwdOp"}:
        key = "input" if op in POOL else "x"
        if p[key][1] != "bfloat16":
            return False, "Native sparse prefill accepts BF16 only; this input dtype cannot be narrowed losslessly."
        if p.get("dtype") not in (None, "bfloat16"):
            return False, "Native weighted-mean output is BF16, so the requested FP32 mean cannot be recovered by casting."
        if p.get("divisor_override") is not None:
            return False, "Native uniform attention divides by the count; a divisor override requires extra numerical rescaling."
        if op in REDUCTION:
            _, _, width, _ = axes_plan(p[key][0], p.get("dim"), p.get("keepdim", False))
            if width == 0:
                return False, "Empty reduction does not have the native attention's empty-row semantics."
        return True, "Select exposed native max_logits/lse, or native uniform weighted mean; gather/pad windows without arithmetic on input values."
    return False, "No equivalent exposed Ascend FlashMLA output within permitted layout, padding, crop and API adaptations."
