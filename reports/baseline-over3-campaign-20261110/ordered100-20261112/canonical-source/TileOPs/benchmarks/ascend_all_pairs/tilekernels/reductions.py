"""Select original mHC partial-sum and top-one outputs."""
from .extended import native, result
from .pointwise import _scale_native


def _layout(case):
    import math
    x = case.inputs[0]
    dim = case.params.get('dim')
    axes = tuple(range(x.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    axes = tuple(d % x.ndim for d in axes)
    kept = tuple(i for i in range(x.ndim) if i not in axes)
    width = math.prod(x.shape[i] for i in axes)
    rows = x.numel() // width
    shape = tuple(1 if i in axes else n for i, n in enumerate(x.shape)) if case.params.get('keepdim') else tuple(x.shape[i] for i in kept)
    return axes, kept, width, rows, shape


def sum_or_mean(case):
    import torch
    fn = native('mhc.norm_fn_kernel', 'mhc_reduce_partials_and_rmsnorm_fwd')
    mean = case._call.signature.name == 'MeanFwdOp'
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd') if mean else None
    axes, kept, width, rows, shape = _layout(case)
    tokens = (rows + 23) // 24
    dtype = getattr(torch, case.params['dtype']) if case.params.get('dtype') is not None else case.inputs[0].dtype
    # This unused normalization auxiliary is input independent, and the selected
    # out_mul output is the raw native reduction, before RMS scaling.
    zeros = torch.zeros((width, tokens, 1), device=case.inputs[0].device, dtype=torch.float32)
    def run(x):
        values = x.permute(*(kept + axes)).contiguous().float().reshape(rows, width).transpose(0, 1)
        partials = torch.nn.functional.pad(values, (0, tokens * 24 - rows)).contiguous().reshape(width, tokens, 1, 24)
        summed = torch.empty((tokens, 1, 24), device=x.device, dtype=torch.float32)
        sumsq = torch.empty((tokens, 1), device=x.device, dtype=torch.float32)
        normalized = torch.empty((tokens, 24), device=x.device, dtype=torch.float32)
        fn(partials, zeros, summed, sumsq, normalized, 1, 1., width)
        values = summed.reshape(-1)[:rows]
        if mean:
            values = _scale_native(torch, product, values, 1. / width)
        return values.to(dtype).reshape(shape)
    return result(run, 'mhc.norm_fn_kernel.mhc_reduce_partials_and_rmsnorm_fwd[out_mul]',
                  'Whole reduction axis maps to the original split axis; pad independent output lanes to native width24; select native raw sum. Mean uses original native multiplication by 1/reduction_size. All input packing inside run.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py')


def minimum_or_boolean(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    name = case._call.signature.name
    if name in ('AllFwdOp', 'AnyFwdOp') and case.inputs[0].dtype != torch.bool:
        raise UnsupportedCase('Boolean API selection requires native bool inputs; nonzero conversion of numerical inputs would add unsupported computation.')
    fn = native('moe.topk_gate_kernel', 'topk_gate')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd') if name != 'AnyFwdOp' else None
    axes, kept, width, rows, shape = _layout(case)
    def run(x):
        packed = x.permute(*(kept + axes)).contiguous().reshape(rows, width)
        scores = packed.float() if product is None else _scale_native(torch, product, packed, -1.)
        index = fn(scores.contiguous(), 1)
        # On Ascend gather bool tensors through their byte representation.
        values = packed.view(torch.uint8).gather(1, index).view(torch.bool) if x.dtype == torch.bool else packed.gather(1, index)
        return values.reshape(shape)
    return result(run, 'moe.topk_gate_kernel.topk_gate + optional mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd',
                  'Native minimum selects Amin/All for bool inputs; native maximum selects Any for bool inputs. Exact widening and gather of original values; whole-axis layout mapping.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py',
                  limitations=('Finite input selection; native large reduction rows can exceed UB.',))
