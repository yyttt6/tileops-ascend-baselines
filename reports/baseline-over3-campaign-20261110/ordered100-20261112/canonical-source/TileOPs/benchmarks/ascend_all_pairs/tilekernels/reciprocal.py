"""Positive reciprocal using original native product and mHC rstd output."""
from .extended import native, result


def reciprocal(case):
    import torch
    reducer = native('mhc.norm_fn_kernel', 'mhc_reduce_partials_and_rmsnorm_fwd')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    def multiply(a, b):
        shape = torch.broadcast_shapes(a.shape, b.shape)
        count = __import__('math').prod(shape)
        left = torch.nn.functional.pad(a.expand(shape).float().contiguous().reshape(-1), (0, -count % 256))
        right = torch.nn.functional.pad(b.expand(shape).float().contiguous().reshape(-1), (0, -count % 256))
        out = torch.empty_like(left).reshape(1, -1)
        product(left.reshape(1, -1), right, out)
        return out.reshape(-1)[:count].reshape(shape)
    def run(x, y=None):
        divisor = x if y is None else y
        count = divisor.numel()
        square = multiply(divisor, divisor).contiguous().reshape(1, count, 1)
        numerator = torch.ones((1, count, 1, 24), device=x.device, dtype=torch.float32)
        raw = torch.empty((count, 1, 24), device=x.device, dtype=torch.float32)
        sq = torch.empty((count, 1), device=x.device, dtype=torch.float32)
        out = torch.empty((count, 24), device=x.device, dtype=torch.float32)
        reducer(numerator, square, raw, sq, out, 1, 0., 1)
        inv = out[:, 0].contiguous().reshape_as(divisor)
        return inv.to(x.dtype) if y is None else multiply(x, inv).to(x.dtype)
    return result(run, 'mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd + mhc_reduce_partials_and_rmsnorm_fwd[out]',
                  'Strictly positive manifest domain: native product forms the square; original one-split mHC rstd output with numerator1 and eps0 supplies reciprocal square root of that square. Native product supplies division. Retains original24-lane layout and complete calls; all value-dependent packing is timed.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py',
                  limitations=('Strictly positive manifest generator domain >=0.5; no floor/trunc division. Original24-lane buffers are materialized.',))
