"""Additional original normalization/rotation output selections."""
from .extended import native, result


def fused_qk_rope(case):
    import torch
    norm = native('quant.norm_forward_kernel', 'norm_forward')
    rope = native('transform.rope_kernel', 'apply_rotary')
    p = case.params
    heads, kv_heads = p['num_heads'], p['num_kv_heads']
    def run(qkv, q_weight, k_weight, cos_sin_cache, positions):
        width = q_weight.numel()
        source = qkv.reshape(qkv.shape[0], heads + 2 * kv_heads, width)
        pieces = []
        for start, count, weight in ((0, heads, q_weight), (heads, kv_heads, k_weight)):
            rows = source[:, start:start+count].float().contiguous().reshape(-1, width)
            normalized = norm(rows, weight.float().contiguous(), p['eps'])[0].to(qkv.dtype)
            rotated = normalized.float().reshape(qkv.shape[0], count, width)
            rope(rotated[..., :cos_sin_cache.shape[-1]], cos_sin_cache.float().contiguous(),
                 positions=positions, interleaved=p['rope_layout'] == 'interleaved')
            pieces.append(rotated.to(qkv.dtype))
        pieces.append(source[:, heads+kv_heads:])
        return torch.cat(pieces, 1).reshape_as(qkv)
    return result(run, 'quant.norm_forward_kernel.norm_forward + transform.rope_kernel.apply_rotary',
                  'Split original Q/K/V layout; original FP32 RMSNorm with weight, manifest dtype rounding, then original in-place RoPE on private widened Q/K. Rejoin unchanged V. Full supplied tables/positions retained; packing per call.',
                  'quant/norm_forward_kernel.py', 'quant/norm_forward_asc.py', 'transform/rope_kernel.py', 'transform/rope_asc.py')


def roots(case):
    import torch
    name = case._call.signature.name
    norm = native('quant.norm_forward_kernel', 'norm_forward')
    reduce = native('mhc.norm_fn_kernel', 'mhc_reduce_partials_and_rmsnorm_fwd')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    from benchmarks.ascend_baselines import UnsupportedCase
    if name == 'DivFwdOp' and case.params.get('rounding_mode') is not None:
        raise UnsupportedCase('Original reciprocal/product route has no floor/trunc output.')
    def run(x, y=None):
        if name in ('ReciprocalFwdOp', 'DivFwdOp'):
            divisor = x if y is None else y
            reciprocal = norm(divisor.float().contiguous().reshape(-1, 1), None, 0., with_rstd=True)[1].reshape_as(divisor)
            if y is None: return reciprocal.to(x.dtype)
            shape = torch.broadcast_shapes(x.shape, y.shape)
            n = __import__('math').prod(shape)
            a = torch.nn.functional.pad(x.expand(shape).float().contiguous().reshape(-1), (0, -n % 256))
            b = torch.nn.functional.pad(reciprocal.expand(shape).contiguous().reshape(-1), (0, -n % 256))
            out = torch.empty_like(a).reshape(1, -1)
            product(a.reshape(1, -1), b, out)
            return out.reshape(-1)[:n].reshape(shape).to(x.dtype)
        count = x.numel()
        values = x.float().contiguous().reshape(1, count, 1)
        numerator = (values[..., None].expand(1, count, 1, 24).contiguous() if name == 'SqrtFwdOp'
                     else torch.ones((1, count, 1, 24), device=x.device, dtype=torch.float32))
        raw = torch.empty((count, 1, 24), device=x.device, dtype=torch.float32)
        sq = torch.empty((count, 1), device=x.device, dtype=torch.float32)
        out = torch.empty((count, 24), device=x.device, dtype=torch.float32)
        reduce(numerator, values, raw, sq, out, 1, 0., 1)
        return out[:, 0].contiguous().reshape_as(x).to(x.dtype)
    return result(run, 'mhc.norm_fn_kernel.mhc_reduce_partials_and_rmsnorm_fwd[out] or quant.norm_forward_kernel.norm_forward[with_rstd] + mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd',
                  'Positive manifest domain: original RMS partial API with one complete split, square-sum argument x, group_size1/eps0 and numerator1 or x selects rsqrt/sqrt. Original width1 RMS rstd selects reciprocal; native pointwise product supplies division. Original24-lane duplication is materialized and timed; no kernel edits.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'quant/norm_forward_kernel.py', 'quant/norm_forward_asc.py',
                  limitations=('Manifest generator supplies strictly positive values; reciprocal via width1 RMS is only valid on this positive domain. Large roots materialize original24-lane buffers.',))


def numerical_boolean(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    from .reductions import minimum_or_boolean, _layout
    from .pointwise import _scale_native
    if case.inputs[0].dtype == torch.bool:
        return minimum_or_boolean(case)
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase('Native numerical nonzero route requires lossless real FP32 input widening.')
    top = native('moe.topk_gate_kernel', 'topk_gate')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    axes, kept, width, rows, shape = _layout(case)
    is_all = case._call.signature.name == 'AllFwdOp'
    def run(x):
        values = x.permute(*(kept + axes)).contiguous().float().reshape(rows, width)
        negative = _scale_native(torch, product, values, -1.)
        paired = torch.stack((values.reshape(-1), negative.reshape(-1)), -1)
        absolute = paired.gather(1, top(paired.contiguous(), 1)).reshape(rows, width)
        order = _scale_native(torch, product, absolute, -1.) if is_all else absolute
        selected = absolute.gather(1, top(order.contiguous(), 1)).reshape(-1)
        # With zero first, native stable top-one index is exactly0/1 for >0.
        bits = top(torch.stack((torch.zeros_like(selected), selected), -1).contiguous(), 1)
        return bits.to(torch.bool).reshape(shape)
    return result(run, 'mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd + moe.topk_gate_kernel.topk_gate',
                  'Native sign reversal/top-one forms absolute magnitude; native min/max reduces whole axis; native stable selection against zero supplies exact boolean index0/1. Every value transform is native and packing is inside run.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py',
                  limitations=('Finite real FP16/BF16/FP32 inputs; full reduction retains original UB limit.',))
