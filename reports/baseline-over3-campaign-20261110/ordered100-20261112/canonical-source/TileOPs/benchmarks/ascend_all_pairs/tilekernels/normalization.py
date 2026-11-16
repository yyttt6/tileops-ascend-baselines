"""Centered normalization composed from original TileKernels outputs."""
from .extended import native, result
from .moments import centered_rows, add_values, product_values, sum_rows


def normalization(case):
    import math
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    name = case._call.signature.name
    p = case.params
    x = case.inputs[0]
    eps = p.get('eps')
    eps = torch.finfo(torch.float32).eps if eps is None else eps
    if name == 'InstanceNormFwdOp' and (case.inputs[1] is not None or not p['use_input_stats']):
        raise UnsupportedCase('Running-stat outputs/update and evaluation route remain unimplemented; no dropped outputs.')
    if name == 'GroupNormFwdOp':
        rows = x.shape[0] * p['num_groups']
        width = x.numel() // rows
    elif name == 'InstanceNormFwdOp':
        rows = x.shape[0] * x.shape[1]
        width = x.numel() // rows
    else:
        width = math.prod(p.get('normalized_shape', (x.shape[-1],)))
        rows = x.numel() // width
    if width > 16384:
        raise UnsupportedCase('Unmodified full-row FP32 native RMS UB domain limited to width <=16384 for this route.')
    reducer = native('mhc.norm_fn_kernel', 'mhc_reduce_partials_and_rmsnorm_fwd')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    norm = native('quant.norm_forward_kernel', 'norm_forward')
    def run(x, *args):
        fused = name == 'FusedAddLayerNormFwdOp'
        if fused:
            residual, weight, bias = args
            summed = add_values(torch, norm, x.float(), residual.float()).to(x.dtype)
            source = summed
        elif name == 'InstanceNormFwdOp':
            _, _, weight, bias = args
            source = x
        else:
            weight, bias = args
            source = x
        values = source.float().contiguous().reshape(rows, width)
        _, centered = centered_rows(torch, reducer, product, norm, values)
        # RMS(centered) is LayerNorm: variance is the population centered second moment.
        if name == 'InstanceNormFwdOp' and width % 256:
            # Use the native complete square sum and RMS API for ragged spatial
            # rows. All original coordinates remain; only independent output
            # lanes are padded to the API's fixed24 width.
            squares = product_values(torch, product, centered, centered)
            square_sum = sum_rows(torch, reducer, squares)
            groups = (width + 23) // 24
            tokens = rows * groups
            numerator = torch.nn.functional.pad(centered, (0, groups * 24 - width)).reshape(1, tokens, 1, 24)
            denominator = square_sum[:, None].expand(rows, groups).contiguous().reshape(1, tokens, 1)
            raw = torch.empty((tokens, 1, 24), device=x.device, dtype=torch.float32)
            sq = torch.empty((tokens, 1), device=x.device, dtype=torch.float32)
            normalized24 = torch.empty((tokens, 24), device=x.device, dtype=torch.float32)
            reducer(numerator, denominator, raw, sq, normalized24, width, eps, 1)
            normalized = normalized24.reshape(rows, groups * 24)[:, :width].contiguous().reshape(x.shape)
        else:
            normalized = norm(centered.contiguous(), None, eps)[0].reshape(x.shape)
        channel = name in ('GroupNormFwdOp', 'InstanceNormFwdOp')
        if weight is not None:
            w = weight.float().reshape((1, x.shape[1]) + (1,) * (x.ndim - 2)) if channel else weight.float()
            normalized = product_values(torch, product, normalized, w)
        if bias is not None:
            b = bias.float().reshape((1, x.shape[1]) + (1,) * (x.ndim - 2)) if channel else bias.float()
            normalized = add_values(torch, norm, normalized, b)
        out = normalized.to(x.dtype)
        return (out, summed) if fused else out
    return result(run, 'mhc complete sum/product + quant.norm_forward[normalized,residual]',
                  'Native complete mean and residual subtraction; native RMS of centered input supplies population normalization. Ragged InstanceNorm uses native complete square sum and original mHC RMS normalization of24-column output groups. Original affine and group/instance axes; FusedAddLayerNorm rounds full residual before normalization and returns both outputs. No native kernel edits or Torch arithmetic.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'quant/norm_forward_kernel.py', 'quant/norm_forward_asc.py', 'quant/rstd.py',
                  limitations=('Finite original manifest inputs; no running-stat InstanceNorm support; full-row native RMS width<=16384.',))
