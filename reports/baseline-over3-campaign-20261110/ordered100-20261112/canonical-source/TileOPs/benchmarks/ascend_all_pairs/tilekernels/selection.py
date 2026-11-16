"""Original native products, residual sums and ordered selections."""
from .extended import native, result


def selection(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    name, p = case._call.signature.name, case.params
    if any(x is not None and x.dtype not in (torch.bool, torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.int16) for x in case.inputs):
        raise UnsupportedCase('Native FP32 comparison/product requires exact input widening; full INT32/INT64 domains are not exactly representable.')
    top = native('moe.topk_gate_kernel', 'topk_gate')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    add = native('quant.norm_forward_kernel', 'norm_forward')
    def multiply(a, b):
        shape = torch.broadcast_shapes(a.shape, b.shape)
        n = __import__('math').prod(shape)
        left = torch.nn.functional.pad(a.expand(shape).float().contiguous().reshape(-1), (0, -n % 256))
        right = torch.nn.functional.pad(b.expand(shape).float().contiguous().reshape(-1), (0, -n % 256))
        out = torch.empty_like(left).reshape(1, -1)
        product(left.reshape(1, -1), right, out)
        return out.reshape(-1)[:n].reshape(shape)
    def constant(x, value):
        return torch.full((), value, dtype=torch.float32, device=x.device)
    def sum_native(a, b):
        shape = torch.broadcast_shapes(a.shape, b.shape)
        n = __import__('math').prod(shape)
        left = torch.nn.functional.pad(a.expand(shape).float().contiguous().reshape(-1), (0, -n % 256)).reshape(-1, 256)
        right = torch.nn.functional.pad(b.expand(shape).float().contiguous().reshape(-1), (0, -n % 256)).reshape(-1, 256)
        return add(left, None, 1e-6, residual=right)[2].reshape(-1)[:n].reshape(shape)
    def choose(a, b, minimum=False):
        shape = torch.broadcast_shapes(a.shape, b.shape)
        pair = torch.stack((a.expand(shape).float().reshape(-1), b.expand(shape).float().reshape(-1)), -1)
        scores = multiply(pair, constant(pair, -1.)) if minimum else pair
        return pair.gather(1, top(scores.contiguous(), 1)).reshape(shape)
    def relation(a, b, strict=False):
        # Prefer the first column on a tie; map selected original column to a bool.
        shape = torch.broadcast_shapes(a.shape, b.shape)
        pair = torch.stack((b.expand(shape).float().reshape(-1), a.expand(shape).float().reshape(-1)), -1) if strict else torch.stack((a.expand(shape).float().reshape(-1), b.expand(shape).float().reshape(-1)), -1)
        ids = top(pair.contiguous(), 1)
        labels = torch.tensor([0., 1.] if strict else [1., 0.], device=a.device, dtype=torch.float32)
        return labels.expand(pair.shape[0], 2).gather(1, ids).reshape(shape)
    def nonzero(x):
        if x.dtype == torch.bool:
            return x.float()
        zero = constant(x, 0.)
        # For finite values, x>0 or 0>x; they cannot both be true.
        return sum_native(relation(x, zero, True), relation(zero, x, True))
    def clamp(x, lo, hi):
        out = x.float()
        if lo is not None: out = choose(out, lo)
        if hi is not None: out = choose(out, hi, True)
        return out
    def run(x, *args):
        xf = x.float()
        if name in ('GeFwdOp', 'GtFwdOp', 'LeFwdOp', 'LtFwdOp', 'EqFwdOp', 'NeFwdOp'):
            y = args[0]
            if name == 'GeFwdOp': out = relation(x, y)
            elif name == 'GtFwdOp': out = relation(x, y, True)
            elif name == 'LeFwdOp': out = relation(y, x)
            elif name == 'LtFwdOp': out = relation(y, x, True)
            else:
                out = multiply(relation(x, y), relation(y, x))
                if name == 'NeFwdOp': out = sum_native(constant(x, 1.), multiply(out, constant(x, -1.)))
            return out.bool()
        if name.startswith('Logical'):
            a = nonzero(x)
            if name == 'LogicalNotFwdOp': out = sum_native(constant(x, 1.), multiply(a, constant(x, -1.)))
            elif name == 'LogicalAndFwdOp': out = multiply(a, nonzero(args[0]))
            else: out = choose(a, nonzero(args[0]))
            return out.bool()
        if name == 'ReluFwdOp':
            out = choose(xf, constant(x, 0.))
        elif name in ('ClampScalarFwdOp', 'HardtanhFwdOp'):
            lo, hi = (p.get('min'), p.get('max')) if name == 'ClampScalarFwdOp' else (p['min_val'], p['max_val'])
            out = clamp(xf, None if lo is None else constant(x, lo), None if hi is None else constant(x, hi))
        elif name == 'ClampTensorFwdOp':
            out = clamp(xf, args[0], args[1])
        elif name == 'LeakyReluFwdOp':
            slope = p['negative_slope']
            out = choose(xf, multiply(xf, constant(x, slope)), minimum=slope > 1.)
        elif name == 'PreluFwdOp':
            weight = args[0].reshape(1, -1, *([1] * (x.ndim - 2)))
            positive = choose(xf, constant(x, 0.))
            negative = choose(xf, constant(x, 0.), True)
            out = sum_native(positive, multiply(negative, weight))
        elif name in ('HardsigmoidFwdOp', 'HardswishFwdOp'):
            out = clamp(sum_native(xf, constant(x, 3.)), constant(x, 0.), constant(x, 6.))
            out = multiply(out, constant(x, 1. / 6.))
            if name == 'HardswishFwdOp': out = multiply(xf, out)
        elif name == 'LerpScalarFwdOp' and p['weight'] == 0.5:
            out = multiply(sum_native(xf, args[0]), constant(x, 0.5))
        elif name in ('LerpScalarFwdOp', 'LerpTensorFwdOp'):
            weight = constant(x, p['weight']) if name == 'LerpScalarFwdOp' else args[1]
            delta = sum_native(args[0], multiply(xf, constant(x, -1.)))
            out = sum_native(xf, multiply(delta, weight))
        else:
            raise UnsupportedCase(name)
        return out.to(x.dtype)
    return result(run, 'moe.topk_gate_kernel.topk_gate + mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd + quant.norm_forward_kernel.norm_forward[2]',
                  'Select/clamp via original stable top-one; native products and residual sums implement requested affine/piecewise formulas. Comparisons map native selected indices to exact0/1 labels; logical operations use native comparisons/products on0/1. Broadcast/packing/gather inside run; no Torch arithmetic.',
                  'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py',
                  'quant/norm_forward_kernel.py', 'quant/norm_forward_asc.py',
                  limitations=('Finite manifest inputs; full INT32/INT64 comparison domains are excluded because exact FP32 widening is unavailable.',))
