"""Original vllm-ascend arithmetic, reduction, normalization and GEMM APIs."""
from math import prod
from pathlib import Path

from .runtime import TRITON_ROOT, candidate, module, validate


def arithmetic(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + "muls_add.py", hardware=True)
    op = case._call.signature.name
    shape = tuple(case.params["output"][0])
    alpha = case.params.get("alpha", 1)

    def run(x, y=None):
        # Flatten to a fixed singleton feature dimension so the ORIGINAL host's
        # block size remains a valid power of two even for ragged input widths.
        xp = x.expand(shape).contiguous().reshape(-1, 1)
        if op == "NegFwdOp":
            return native.muls_add_triton(xp, torch.zeros_like(xp), -1.0).reshape(shape)
        yp = y.expand(shape).contiguous().reshape(-1, 1)
        scale = -alpha if op == "SubFwdOp" else alpha
        return native.muls_add_triton(yp, xp, scale).reshape(shape)

    return candidate(run, "muls_add_triton", "Per-call broadcast/contiguous singleton-column view; original scale*x+y API implements add/subtract or negation; no input cache.", [Path(__file__), *sources])


def mean(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + "batch_invariant/mean.py")
    p = case.params
    shape = tuple(p["output"][0])
    dims = p.get("dim")
    rank = len(p["x"][0])
    dims = tuple(range(rank)) if dims is None or dims == [] else (dims if isinstance(dims, (tuple, list)) else (dims,))
    dims = tuple(d % rank for d in dims)
    order = tuple(d for d in range(rank) if d not in dims) + dims
    dtype = p.get("dtype") or p["output"][1]

    def run(x):
        width = prod(x.shape[d] for d in dims)
        packed = x.permute(order).contiguous().reshape(-1, width)
        # FP16/BF16 -> FP32 is exact. Bypass the public multi-dim Torch fallback
        # and its single-dim default-FP16 bug by selecting mean_dim explicitly.
        result = native.mean_dim(packed.float(), 1, dtype=torch.float32)
        return result.to(getattr(torch, dtype)).reshape(shape)

    return candidate(run, "mean_dim", "Move/flatten all reduction axes to one axis; exactly widen FP16/BF16 to FP32 for native accumulation, cast original output dtype and restore keepdim shape; no torch reduction.", [Path(__file__), *sources])


def mean_pooling(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + "batch_invariant/mean.py")
    chunk = case.params["chunk_size"]

    def run(x, offsets=None, indices=None):
        if offsets is None:
            slices = [(i, min(i + chunk, x.shape[1])) for i in range(0, x.shape[1], chunk)]
        else:
            # Metadata transfers and slicing are inside the measured call.
            bounds = offsets.tolist()
            slices = [(bounds[s] + c * chunk, min(bounds[s] + (c + 1) * chunk, bounds[s + 1]))
                      for s, c in indices.tolist()]
        return torch.stack([native.mean_dim(x[:, begin:end].float().contiguous(), 1,
                                           dtype=torch.float32).to(x.dtype)
                            for begin, end in slices], dim=1)

    return candidate(run, "mean_dim (per sequence chunk)", "Original mean on each complete chunk, including ragged tail divisor; all metadata reads, packing, widening and stack within run.", [Path(__file__), *sources], timing_mode="host_completed")


def normalization(case):
    validate(case)
    import torch
    op = case._call.signature.name
    rms = op in ("RMSNormFwdOp", "FusedAddRMSNormFwdOp")
    fused = op.startswith("FusedAdd")
    native, sources = module(TRITON_ROOT + ("batch_invariant/rmsnorm.py" if rms else "layernorm_gated.py"))
    if fused:
        addition, add_sources = module(TRITON_ROOT + "muls_add.py", hardware=True)
        sources += add_sources
    p = case.params
    eps = p.get("eps")
    eps = 2 ** -23 if eps is None else eps
    hidden = prod(p.get("normalized_shape", [p["x"][0][-1]]))

    def normalize(x, w, b=None):
        packed = x.contiguous().reshape(-1, hidden)
        weight = torch.ones(hidden, device=x.device, dtype=x.dtype) if w is None else w.contiguous().reshape(hidden)
        if rms:
            return native.rms_norm(packed, weight, eps).reshape(x.shape)
        bias = b.contiguous().reshape(hidden) if b is not None else None
        return native.layer_norm_fwd_npu(packed, weight, bias, eps)[0].reshape(x.shape)

    def run(x, *args):
        if fused:
            residual, w, *bias = args
            # Manifest rounds the residual sum BEFORE normalization. Its two
            # outputs must not be replaced by an unrounded fused norm result.
            summed = addition.muls_add_triton(x.contiguous().reshape(-1, 1),
                                             residual.contiguous().reshape(-1, 1), 1.0).reshape(x.shape)
            return normalize(summed, w, bias[0] if bias else None), summed
        w, *bias = args
        return normalize(x, w, bias[0] if bias else None)

    return candidate(run, ("muls_add_triton + " if fused else "") + ("rms_norm" if rms else "layer_norm_fwd_npu"), "Collapse normalized dimensions; native weighted normalization; optional affine disabled with unit weight. Fused cases first call original native addition and retain its input-dtype rounded sum as the second output.", [Path(__file__), *sources])


def adaptive_layer_norm(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + "layernorm_gated.py")
    eps = case.params.get("eps", 1e-5)

    def run(x, scale, shift):
        n = x.shape[-1]
        xp, sp, bp = (a.contiguous().reshape(-1, n) for a in (x, scale, shift))
        out = [native.layer_norm_fwd_npu(xp[i:i+1], sp[i], bp[i], eps)[0] for i in range(len(xp))]
        return torch.cat(out).reshape(x.shape)

    return candidate(run, "layer_norm_fwd_npu (row-specific affine)", "Per-row original LayerNorm calls select the corresponding manifest affine scale and shift; concatenate without host arithmetic.", [Path(__file__), *sources])


def group_norm(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + "layernorm_gated.py")
    p = case.params
    instance = case._call.signature.name == "InstanceNormFwdOp"
    groups = p["x"][0][1] if instance else p["num_groups"]
    eps = p.get("eps", 1e-5)

    def run(x, *args):
        if instance:
            _rm, _rv, weight, bias = args
        else:
            weight, bias = args
        batch, channels = x.shape[:2]
        spatial = prod(x.shape[2:])
        width = channels * spatial
        w = torch.ones(channels, device=x.device, dtype=x.dtype) if weight is None else weight
        w = w.reshape(channels, 1).expand(channels, spatial).contiguous().reshape(width)
        b = None if bias is None else bias.reshape(channels, 1).expand(channels, spatial).contiguous().reshape(width)
        packed = x.contiguous().reshape(batch, width)
        out = native.layer_norm_fwd_npu(packed, w, b, eps, group_size=width // groups)[0]
        return out.reshape(x.shape)

    return candidate(run, "layer_norm_fwd_npu(group_size=group_elements)", "Group contiguous channels/spatial elements; repeat per-channel affine weights over spatial positions each call. Input-statistics InstanceNorm is GroupNorm with one group per channel.", [Path(__file__), *sources])


def gemm(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + "batch_invariant/matmul.py")
    op, p = case._call.signature.name, case.params

    def run(a, b):
        if op == "BmmFwdOp":
            return torch.stack([native.matmul_persistent(a[i].contiguous(), b[i].contiguous())
                                for i in range(a.shape[0])])
        aa = a.transpose(-1, -2) if p.get("trans_a", False) else a
        bb = b.transpose(-1, -2) if p.get("trans_b", False) else b
        return native.matmul_persistent(aa.contiguous(), bb.contiguous())

    return candidate(run, "matmul_persistent", "Honor transpose flags; contiguous views. BMM dispatches one original complete GEMM per batch and stacks outputs inside run; original FP32 accumulation and fixed tiles unchanged.", [Path(__file__), *sources])


def grouped_gemm(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + "batch_invariant/matmul.py")
    ta, tb = case.params["transpose_a"], case.params["transpose_b"]

    def run(a, b, sizes, offsets):
        # Both current metadata inputs are consumed each invocation.
        counts, starts = sizes.tolist(), offsets.tolist()
        outputs = []
        for i, (size, start) in enumerate(zip(counts, starts, strict=True)):
            aa = a[start:start+size]
            if ta:
                aa = aa.transpose(0, 1)
                bb = b[:, start:start+size].transpose(0, 1) if tb else b[start:start+size]
            else:
                bb = b[i].transpose(0, 1) if tb else b[i]
            if aa.shape[0] == 0:
                outputs.append(torch.empty((0, bb.shape[1]), dtype=a.dtype, device=a.device))
            else:
                outputs.append(native.matmul_persistent(aa.contiguous(), bb.contiguous()))
        return torch.stack(outputs) if ta else torch.cat(outputs)

    return candidate(run, "matmul_persistent (segmented)", "Read current group sizes and offsets inside run; slice each complete GEMM, honor all transpose combinations, concatenate or stack original outputs.", [Path(__file__), *sources], timing_mode="host_completed")


def pointwise_conv(case):
    validate(case)
    import torch
    import torch.nn.functional as F
    native, sources = module(TRITON_ROOT + "batch_invariant/matmul.py")
    p = case.params
    rank = len(p["input"][0]) - 2
    def vector(v):
        return (v,) * rank if isinstance(v, int) else tuple(v)
    padding = (0,) * rank if isinstance(p["padding"], str) else vector(p["padding"])
    stride = vector(p["stride"])
    groups = p["groups"]

    def run(x, weight, bias=None):
        data = F.pad(x, tuple(v for pad in reversed(padding) for v in (pad, pad)))
        data = data[(slice(None), slice(None), *(slice(None, None, s) for s in stride))]
        spatial = data.shape[2:]
        channels = x.shape[1] // groups
        out_channels = weight.shape[0] // groups
        outputs = []
        for group in range(groups):
            inp = data[:, group*channels:(group+1)*channels]
            a = inp.permute(0, *range(2, inp.ndim), 1).contiguous().reshape(-1, channels)
            w = weight[group*out_channels:(group+1)*out_channels].reshape(out_channels, channels)
            b = None if bias is None else bias[group*out_channels:(group+1)*out_channels]
            output = native.matmul_persistent(a, w.transpose(0, 1).contiguous(), bias=b)
            outputs.append(output.reshape(x.shape[0], *spatial, out_channels))
        return torch.cat(outputs, dim=-1).movedim(-1, 1).contiguous()

    return candidate(run, "matmul_persistent (pointwise convolution)", "Only unit spatial kernels: pad/slice spatial coordinates, separate groups, pack channels as GEMM reduction dimension, use original bias API and restore NCHW/NCDHW layout. No im2col or changed kernel math.", [Path(__file__), *sources])


def cumsum(case):
    validate(case)
    import torch.nn.functional as F
    from .runtime import kda_module
    import importlib
    package, sources = kda_module()
    native = importlib.import_module(package.__package__ + ".cumsum")
    dim = case.params["dim"] % len(case.params["x"][0])

    def run(x):
        moved = x.movedim(dim, -1).contiguous()
        width = moved.shape[-1]
        padded = 1 << (width-1).bit_length()
        data = F.pad(moved.reshape(-1, width), (0, padded-width)).reshape(-1, padded, 1, 1)
        out = native.chunk_local_cumsum_vector(data.float(), chunk_size=padded, output_dtype=None)
        return out[..., 0, 0][:, :width].reshape(moved.shape).movedim(-1, dim).contiguous().to(x.dtype)

    return candidate(run, "kda.cumsum.chunk_local_cumsum_vector", "Move each complete scan axis to one native vector chunk; zero-pad to next power of two and crop, with exact FP32 widening. No cross-chunk prefix algorithm or schedule edits.", [Path(__file__), *sources], limitations=("Original vector path materializes a full chunk-by-chunk triangular matrix in each program; manifest widths 4096-32768 may exceed compiler/device resources. No compile or device claim is made.",))


def fused_qk_norm_rope(case):
    validate(case)
    import torch
    norm, sources = module(TRITON_ROOT + "batch_invariant/rmsnorm.py")
    rotation, extra = module(TRITON_ROOT + "rope.py", hardware=True)
    p = case.params

    def run(qkv, q_weight, k_weight, tables, positions):
        heads, kh, width = p["num_heads"], p["num_kv_heads"], q_weight.numel()
        packed = qkv.clone().reshape(qkv.shape[0], heads+2*kh, width)
        parts = []
        for start, count, weight in ((0, heads, q_weight), (heads, kh, k_weight)):
            normalized = norm.rms_norm(packed[:, start:start+count].contiguous().reshape(-1, width).float(),
                                       weight.contiguous(), p["eps"]).reshape(-1, count, width)
            parts.append(rotation.rope_forward_triton_siso(normalized,
                cos_sin_cache=tables.contiguous(), positions=positions.contiguous(),
                rope_dim=tables.shape[-1], is_neox_style=p["rope_layout"] == "neox").to(qkv.dtype))
        return torch.cat((*parts, packed[:, heads+kh:]), dim=1).reshape(qkv.shape)

    return candidate(run, "rms_norm + rope_forward_triton_siso (complete QKV)",
        "Split Q/K/V layout and widen Q/K exactly to FP32; retain the original RMSNorm output in FP32 through original table-indexed RoPE, then round once to the input dtype. Return full QKV with V copied exactly and protect input QKV.", [Path(__file__), *sources, *extra])
