"""Native condition selection, sign/count and affine sigmoid output selection."""
from .extended import native, result
from .pointwise import _scale_native
from .reductions import _layout


def extra(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    name = case._call.signature.name
    top = native('moe.topk_gate_kernel', 'topk_gate')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    norm = native('quant.norm_forward_kernel', 'norm_forward')
    reducer = native('mhc.norm_fn_kernel', 'mhc_reduce_partials_and_rmsnorm_fwd')
    mix = native('mhc.pre_split_mixes_kernel', 'mhc_pre_split_mixes_fwd') if name == 'TanhFwdOp' else None
    if name == 'CountNonzeroFwdOp':
        if case.inputs[0].dtype not in (torch.bool, torch.float16, torch.bfloat16, torch.float32):
            raise UnsupportedCase('This exact native count route requires bool or real FP32-representable input domains.')
        axes, kept, width, rows, count_shape = _layout(case)
        if width > 2**24:
            raise UnsupportedCase('FP32 raw-sum count is not exact for all counts beyond2**24.')
    def add(a, b):
        count = a.numel()
        left = torch.nn.functional.pad(a.float().contiguous().reshape(-1), (0, -count % 256)).reshape(-1, 256)
        right = torch.nn.functional.pad(b.expand_as(a).float().contiguous().reshape(-1), (0, -count % 256)).reshape(-1, 256)
        return norm(left, None, 1e-6, residual=right)[2].reshape(-1)[:count].reshape_as(a)
    def positive(x):
        rows = torch.stack((torch.zeros_like(x).reshape(-1), x.reshape(-1)), -1)
        return top(rows.contiguous(), 1).float().reshape_as(x)
    def run(*inputs):
        x = inputs[0]
        if name in ('WhereFwdOp', 'MaskedFillScalarFwdOp', 'MaskedFillTensorFwdOp'):
            if name == 'WhereFwdOp': condition, right, left = inputs
            else:
                left, condition = inputs[:2]
                right = (torch.full((), case.params['value'], device=left.device, dtype=left.dtype)
                         if name == 'MaskedFillScalarFwdOp' else inputs[2])
            shape = torch.broadcast_shapes(condition.shape, left.shape, right.shape)
            cond = condition.expand(shape).float().reshape(-1)
            idx = top(torch.stack((torch.zeros_like(cond), cond), -1).contiguous(), 1)
            pair = torch.stack((left.expand(shape).reshape(-1), right.expand(shape).reshape(-1)), -1)
            return pair.gather(1, idx).reshape(shape)
        if name == 'TanhFwdOp':
            count = x.numel()
            rows4 = torch.nn.functional.pad(x.float().contiguous().reshape(-1), (0, -count % 256)).reshape(-1, 4)
            packed = torch.nn.functional.pad(rows4, (4, 16))
            pre, post = torch.empty_like(rows4), torch.empty_like(rows4)
            comb = torch.empty((rows4.shape[0], 16), device=x.device, dtype=torch.float32)
            scale = torch.tensor([1., 2., 1.], device=x.device, dtype=torch.float32)
            bias = torch.zeros(24, device=x.device, dtype=torch.float32)
            mix(packed, scale, bias, pre, post, comb, 2., 0.)
            values = post.reshape(-1)[:count].reshape_as(x)
            return add(values, torch.full((), -1., device=x.device, dtype=torch.float32)).to(x.dtype)
        xf = x.float()
        negative = _scale_native(torch, product, xf, -1.)
        if name == 'SignFwdOp':
            return add(positive(xf), _scale_native(torch, product, positive(negative), -1.)).to(x.dtype)
        paired = torch.stack((xf.reshape(-1), negative.reshape(-1)), -1)
        absolute = paired.gather(1, top(paired.contiguous(), 1)).reshape_as(x)
        bits = positive(absolute)
        values = bits.permute(*(kept + axes)).contiguous().reshape(rows, width).transpose(0, 1)
        tokens = (rows + 23) // 24
        partials = torch.nn.functional.pad(values, (0, tokens * 24 - rows)).contiguous().reshape(width, tokens, 1, 24)
        auxiliary = torch.zeros((width, tokens, 1), device=x.device, dtype=torch.float32)
        raw = torch.empty((tokens, 1, 24), device=x.device, dtype=torch.float32)
        sq = torch.empty((tokens, 1), device=x.device, dtype=torch.float32)
        out = torch.empty((tokens, 24), device=x.device, dtype=torch.float32)
        reducer(partials, auxiliary, raw, sq, out, 1, 1., width)
        return raw.reshape(-1)[:rows].reshape(count_shape).to(torch.int64)
    if name in ('WhereFwdOp', 'MaskedFillScalarFwdOp', 'MaskedFillTensorFwdOp'):
        api = 'moe.topk_gate_kernel.topk_gate(k=1) + gather'
    elif name == 'TanhFwdOp':
        api = 'mhc.pre_split_mixes_kernel.mhc_pre_split_mixes_fwd[post] + quant.norm_forward_kernel.norm_forward[2]'
    elif name == 'SignFwdOp':
        api = 'moe.topk_gate_kernel.topk_gate + mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd + quant.norm_forward_kernel.norm_forward[2]'
    else:
        api = 'moe.topk_gate_kernel.topk_gate + mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd + mhc.norm_fn_kernel.mhc_reduce_partials_and_rmsnorm_fwd[out_mul]'
    return result(run, api,
                  'Original native stable top-one chooses the branch or exact positive/nonzero indicator; gather preserves original output bits. Sign uses native difference of indicators; count uses original full raw split sum of exact0/1. Tanh selects native2*sigmoid(2*x) post output then native residual subtraction1. All value transforms and packing occur per call.',
                  'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py', 'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py',
                  'mhc/pre_split_mixes_kernel.py', 'mhc/pre_split_mixes_asc.py', 'quant/norm_forward_kernel.py', 'quant/norm_forward_asc.py',
                  limitations=('Finite sign/count inputs; count rows are at most2**24 values and inputs must widen exactly to FP32.',))
