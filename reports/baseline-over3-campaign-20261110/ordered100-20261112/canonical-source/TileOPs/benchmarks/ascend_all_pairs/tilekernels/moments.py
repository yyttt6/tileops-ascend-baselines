"""Complete mean and centered moments using unmodified native API outputs."""
from .extended import native, result
from .pointwise import _scale_native
from .reductions import _layout


def sum_rows(torch, reducer, values):
    rows, width = values.shape
    tokens = (rows + 23) // 24
    parts = torch.nn.functional.pad(values.transpose(0, 1), (0, tokens * 24 - rows)).contiguous().reshape(width, tokens, 1, 24)
    aux = torch.zeros((width, tokens, 1), device=values.device, dtype=torch.float32)
    raw = torch.empty((tokens, 1, 24), device=values.device, dtype=torch.float32)
    sq = torch.empty((tokens, 1), device=values.device, dtype=torch.float32)
    out = torch.empty((tokens, 24), device=values.device, dtype=torch.float32)
    reducer(parts, aux, raw, sq, out, 1, 1., width)
    return raw.reshape(-1)[:rows]


def product_values(torch, product, x, y):
    count = x.numel()
    left = torch.nn.functional.pad(x.contiguous().reshape(-1), (0, -count % 256))
    right = torch.nn.functional.pad(y.expand_as(x).contiguous().reshape(-1), (0, -count % 256))
    out = torch.empty_like(left).reshape(1, -1)
    product(left.reshape(1, -1), right, out)
    return out.reshape(-1)[:count].reshape_as(x)


def add_values(torch, norm, x, y):
    # Only select the native residual output; chunking cannot alter elementwise sum.
    count = x.numel()
    left = torch.nn.functional.pad(x.contiguous().reshape(-1), (0, -count % 256)).reshape(-1, 256)
    right = torch.nn.functional.pad(y.expand_as(x).contiguous().reshape(-1), (0, -count % 256)).reshape(-1, 256)
    return norm(left, None, 1e-6, residual=right)[2].reshape(-1)[:count].reshape_as(x)


def centered_rows(torch, reducer, product, norm, values):
    means = _scale_native(torch, product, sum_rows(torch, reducer, values), 1. / values.shape[1])
    negative = _scale_native(torch, product, means, -1.)
    return means, add_values(torch, norm, values, negative[:, None])


def sqrt_values(torch, reducer, top, value):
    rows = value.numel()
    index = top(torch.stack((torch.zeros_like(value), value), -1).contiguous(), 1)
    denominator = torch.stack((torch.ones_like(value), value), -1).gather(1, index).reshape(1, rows, 1).contiguous()
    numerator = value.reshape(1, rows, 1, 1).expand(1, rows, 1, 24).contiguous()
    raw = torch.empty((rows, 1, 24), device=value.device, dtype=torch.float32)
    sq = torch.empty((rows, 1), device=value.device, dtype=torch.float32)
    out = torch.empty((rows, 24), device=value.device, dtype=torch.float32)
    reducer(numerator, denominator, raw, sq, out, 1, 0., 1)
    return out[:, 0]


def moments(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    axes, kept, width, rows, shape = _layout(case)
    correction = case.params.get('correction')
    correction = 1 if correction is None else correction
    if width <= correction:
        raise UnsupportedCase('Native moment composition requires positive corrected divisor.')
    reducer = native('mhc.norm_fn_kernel', 'mhc_reduce_partials_and_rmsnorm_fwd')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    norm = native('quant.norm_forward_kernel', 'norm_forward')
    top = native('moe.topk_gate_kernel', 'topk_gate')
    name = case._call.signature.name
    def run(x):
        values = x.float().permute(*(kept + axes)).contiguous().reshape(rows, width)
        means, centered = centered_rows(torch, reducer, product, norm, values)
        squares = product_values(torch, product, centered, centered)
        variance = _scale_native(torch, product, sum_rows(torch, reducer, squares), 1. / (width - correction))
        value = sqrt_values(torch, reducer, top, variance) if name == 'StdFwdOp' else variance
        value = value.to(x.dtype).reshape(shape)
        return (value, means.to(x.dtype).reshape(shape)) if name == 'VarMeanFwdOp' else value
    return result(run, 'mhc complete sum/product + quant.norm_forward[residual] + optional native RMS sqrt',
                  'Whole original axes mapped to full split sum. Native mean, subtraction, centered square sum and correction scaling; Std selects native sqrt. VarMean returns both full original outputs. FP32 widening and final original dtype; unchanged oracle/tolerance.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'quant/norm_forward_kernel.py', 'quant/norm_forward_asc.py', 'quant/rstd.py', 'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py')
