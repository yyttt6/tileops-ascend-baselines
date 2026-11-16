"""BF16 reductions expressed by the original GEMM contraction and Alpha API.

Only axis packing, a constant all-one operand and declared output layout are
adapted. All sums and mean scaling execute in the unchanged native GEMM.
"""
from math import prod
from .adapters import _bf16, _device, _nt_operands, _unsupported
from .runtime import candidate, load_native


def reduction_plan(params):
    shape = tuple(params['x'][0])
    rank = len(shape)
    dim = params.get('dim')
    axes = tuple(range(rank)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    if not axes:
        axes = tuple(range(rank))
    if not rank or min(shape) <= 0:
        raise ValueError('Native reduction requires a nonempty nonscalar input.')
    if any(not -rank <= axis < rank for axis in axes):
        raise ValueError('Reduction axis out of range.')
    axes = tuple(axis % rank for axis in axes)
    if len(set(axes)) != len(axes):
        raise ValueError('Duplicate reduction axes.')
    kept = tuple(axis for axis in range(rank) if axis not in axes)
    width = prod(shape[axis] for axis in axes)
    output = tuple(1 if axis in axes else n for axis, n in enumerate(shape)) if params.get('keepdim', False) else tuple(shape[axis] for axis in kept)
    if tuple(params['output'][0]) != output:
        raise ValueError('Output shape differs from the original reduction contract.')
    if params['x'][1] != 'bfloat16' or params['output'][1] not in ('bfloat16', 'float32'):
        raise ValueError('The original GEMM accepts BF16 operands and BF16/FP32 output; no lossy input narrowing.')
    return kept + axes, prod(shape) // width, width, output


def reduction(case):
    op = case._call.signature.name
    if op not in ('SumFwdOp', 'MeanFwdOp'):
        _unsupported('Only Sum and Mean map to this original contraction/Alpha API.')
    try:
        order, rows, width, output_shape = reduction_plan(case.params)
    except ValueError as exc:
        _unsupported(str(exc))
    x, = case.inputs
    _bf16(x)
    _device(x)
    _, functions, hashes = load_native('bf16_gemm_nt')
    native = functions['bf16_gemm_nt']
    output_dtype = case.params['output'][1]

    def run(x):
        import torch
        lhs = x.permute(order).contiguous().reshape(rows, width)
        rhs = torch.ones((1, width), dtype=x.dtype, device=x.device)
        lhs, rhs = _nt_operands(lhs, rhs)
        output = torch.empty((rows, 1), dtype=getattr(torch, output_dtype), device=x.device)
        if op == 'MeanFwdOp':
            native(lhs, rhs, output, alpha=1.0 / width)
        else:
            native(lhs, rhs, output)
        return output.reshape(output_shape)

    return candidate(run, 'deep_gemm.bf16_gemm_nt(alpha=1/K for Mean)',
                     'Permute/fold the complete reduction axes; exact BF16 one-vector operand; original FP32 accumulation and native scalar Alpha for Mean; original declared output dtype/shape. No split K, target torch arithmetic or kernel changes.', hashes)
