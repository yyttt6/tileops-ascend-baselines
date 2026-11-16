"""Select existing native affine/selection outputs without changing kernels."""
from .extended import native, result


def _scale_native(torch, fn, x, scalar):
    count = x.numel()
    flat = torch.nn.functional.pad(x.float().contiguous().reshape(-1), (0, -count % 256))
    weights = torch.full_like(flat, scalar)
    out = torch.empty_like(flat).reshape(1, -1)
    fn(flat.reshape(1, -1), weights, out)
    return out.reshape(-1)[:count].reshape_as(x)


def neg(case):
    import torch
    fn = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    def run(x):
        return _scale_native(torch, fn, x, -1.).to(x.dtype)
    return result(run, 'mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd',
                  'Select original pointwise product with constant -1 weight; flatten, exact FP32 widening and output dtype restoration.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py')


def sub(case):
    import torch
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    norm = native('quant.norm_forward_kernel', 'norm_forward')
    shape = torch.broadcast_shapes(*(x.shape for x in case.inputs))
    alpha = case.params.get('alpha', 1)
    count = __import__('math').prod(shape)
    def run(x, y):
        negative = _scale_native(torch, product, y.expand(shape), -alpha)
        left = x.expand(shape).contiguous().float().reshape(-1)
        right = negative.contiguous().reshape(-1)
        left = torch.nn.functional.pad(left, (0, -count % 256)).reshape(-1, 256)
        right = torch.nn.functional.pad(right, (0, -count % 256)).reshape(-1, 256)
        summed = norm(left, None, 1e-6, residual=right)[2]
        return summed.reshape(-1)[:count].to(x.dtype).reshape(shape)
    return result(run, 'mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd + quant.norm_forward_kernel.norm_forward[2]',
                  'Original native multiplication by -alpha then original native residual sum; exact input widening, broadcast/packing and output restoration. Both complete kernels counted.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'quant/norm_forward_kernel.py', 'quant/norm_forward_asc.py')


def sigmoid(case):
    import torch
    fn = native('mhc.pre_split_mixes_kernel', 'mhc_pre_split_mixes_fwd')
    x = case.inputs[0]
    scale = torch.ones(3, device=x.device, dtype=torch.float32)
    bias = torch.zeros(24, device=x.device, dtype=torch.float32)
    def run(x):
        # The pre-mix output is sigmoid(x*scale[0]+bias)+pre_eps.
        count = x.numel()
        flat = torch.nn.functional.pad(x.float().contiguous().reshape(-1), (0, -count % 256))
        rows = flat.reshape(-1, 4)
        packed = torch.nn.functional.pad(rows, (0, 20))
        pre = torch.empty_like(rows)
        post = torch.empty_like(rows)
        comb = torch.empty((rows.shape[0], 16), device=x.device, dtype=torch.float32)
        fn(packed, scale, bias, pre, post, comb, 1., 0.)
        return pre.reshape(-1)[:count].to(x.dtype).reshape_as(x)
    return result(run, 'mhc.pre_split_mixes_kernel.mhc_pre_split_mixes_fwd',
                  'Identity affine parameters select native sigmoid pre-mix output; four values per original mHC row with zero padding; all native outputs allocated and full call timed.',
                  'mhc/pre_split_mixes_kernel.py', 'mhc/pre_split_mixes_asc.py')


def maximum(case):
    import torch
    fn = native('moe.topk_gate_kernel', 'topk_gate')
    shape = torch.broadcast_shapes(*(x.shape for x in case.inputs))
    def run(x, y):
        paired = torch.stack((x.expand(shape).reshape(-1), y.expand(shape).reshape(-1)), -1)
        index = fn(paired.float().contiguous(), 1)
        return paired.gather(1, index).reshape(shape)
    return result(run, 'moe.topk_gate_kernel.topk_gate(k=1) + gather',
                  'Broadcast two inputs into two-column rows; exact widening for native maximum selection, gather original value bits.',
                  'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py',
                  limitations=('Finite manifest inputs only: native topk rejects NaN/Inf.',))


def minimum(case):
    import torch
    top = native('moe.topk_gate_kernel', 'topk_gate')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    shape = torch.broadcast_shapes(*(x.shape for x in case.inputs))
    def run(x, y):
        paired = torch.stack((x.expand(shape).reshape(-1), y.expand(shape).reshape(-1)), -1)
        negative = _scale_native(torch, product, paired, -1.)
        return paired.gather(1, top(negative.contiguous(), 1)).reshape(shape)
    return result(run, 'mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd + moe.topk_gate_kernel.topk_gate',
                  'Native sign reversal followed by original maximum-index selection selects the minimum; gather unmodified original values.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py',
                  limitations=('Finite manifest inputs only: native topk rejects NaN/Inf.',))


def abs_value(case):
    import torch
    top = native('moe.topk_gate_kernel', 'topk_gate')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    def run(x):
        negative = _scale_native(torch, product, x, -1.)
        paired = torch.stack((torch.zeros_like(negative).reshape(-1), x.float().reshape(-1), negative.reshape(-1)), -1)
        return paired.gather(1, top(paired.contiguous(), 1)).reshape_as(x).to(x.dtype)
    return result(run, 'mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd + moe.topk_gate_kernel.topk_gate',
                  'Native product by -1 and native maximum selection between x and -x; FP32 exact input representation and final dtype restoration.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py',
                  limitations=('Finite manifest inputs only: native topk rejects NaN/Inf.',))


def argmin(case):
    import torch
    import math
    top = native('moe.topk_gate_kernel', 'topk_gate')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    x = case.inputs[0]
    dim = case.params.get('dim')
    axes = tuple(range(x.ndim)) if dim is None else (dim % x.ndim,)
    kept = tuple(i for i in range(x.ndim) if i not in axes)
    width = math.prod(x.shape[i] for i in axes)
    shape = tuple(1 if i in axes else n for i, n in enumerate(x.shape)) if case.params.get('keepdim') else tuple(x.shape[i] for i in kept)
    def run(x):
        packed = x.permute(*(kept + axes)).contiguous().reshape(-1, width)
        negative = _scale_native(torch, product, packed, -1.)
        return top(negative.contiguous(), 1).reshape(shape)
    return result(run, 'mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd + moe.topk_gate_kernel.topk_gate',
                  'Native product by -1 reverses order exactly; original stable top-one indices; permute whole reduction axis and restore index shape.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py',
                  limitations=('Finite manifest inputs only: native topk rejects NaN/Inf.',))


def softplus(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.params.get('beta', 1.) != 1. or case.params.get('threshold', 20.) != 20.:
        raise UnsupportedCase('This native sqrtsoftplus score adapter supports manifest beta=1, threshold=20.')
    gate = native('moe.moe_topk_gate_forward_kernel', 'moe_topk_gate_forward')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    def run(x):
        count = x.numel()
        rows = torch.nn.functional.pad(x.float().contiguous().reshape(-1), (0, -count % 256)).reshape(-1, 256)
        scores = torch.empty_like(rows)
        gate(rows, 1, False, 0, 1., 0, scoring_func='sqrtsoftplus', scores=scores)
        flat = scores.reshape(-1)
        out = torch.empty_like(flat).reshape(1, -1)
        product(flat.reshape(1, -1), flat, out)
        return out.reshape(-1)[:count].to(x.dtype).reshape_as(x)
    return result(run, 'moe.moe_topk_gate_forward_kernel.moe_topk_gate_forward[scores] + mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd',
                  'Select original sqrt(softplus) score output and square through original native weight-product API; exact widening and output restoration; full routing call counted.',
                  'moe/moe_topk_gate_forward_kernel.py', 'moe/moe_topk_gate_forward_asc.py', 'moe/sqrt_softplus.py',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py',
                  limitations=('Finite FP16/BF16 manifest cases with beta=1 and threshold=20.',))
