"""Exact window/chunk packing for the original BF16 GEMM and Alpha API.

No input arithmetic is performed by Torch: values are gathered, padded and
copied. Each complete window is one original GEMM contraction; its original
pooling divisor is passed to the unchanged native Alpha epilogue.
"""
from itertools import product
from math import prod

from .adapters import _bf16, _device, _nt_operands, _unsupported
from .runtime import candidate, load_native


def _axes(value, rank, default=None):
    if value is None:
        value = default
    values = (value,) * rank if isinstance(value, int) else tuple(value)
    if len(values) != rank:
        raise ValueError('Pool parameter rank mismatch')
    return values


def pool_plan(op, params):
    shape, dtype = params['input']
    output, output_dtype = params['output']
    if dtype != 'bfloat16' or output_dtype != dtype:
        raise ValueError('Original GEMM accepts BF16 pooling values; narrowing other inputs is not lossless')
    spatial, out = tuple(shape[2:]), tuple(output[2:])
    rank = len(spatial)
    if rank not in (1, 2, 3) or tuple(shape[:2]) != tuple(output[:2]) or min(shape) <= 0:
        raise ValueError('Expected nonempty batched NC-spatial original tensors')
    adaptive = op == 'AdaptiveAvgPool2dFwdOp'
    if adaptive:
        if rank != 2:
            raise ValueError('AdaptiveAvgPool2d requires two spatial axes')
        wanted = _axes(params['output_size'], rank, spatial)
        wanted = tuple(n if v is None else v for v, n in zip(wanted, spatial))
    else:
        if op != f'AvgPool{rank}dFwdOp':
            raise ValueError('Unknown average pooling contract')
        kernel = _axes(params['kernel_size'], rank)
        stride = _axes(params.get('stride'), rank, kernel)
        padding = _axes(params.get('padding', 0), rank)
        wanted = []
        for n, k, s, p in zip(spatial, kernel, stride, padding):
            size = (n + 2*p - k + (s-1 if params.get('ceil_mode') else 0)) // s + 1
            if params.get('ceil_mode') and (size-1)*s >= n+p:
                size -= 1
            wanted.append(size)
        wanted = tuple(wanted)
    if tuple(wanted) != out or min(out) <= 0:
        raise ValueError('Pool geometry differs from the original output shape')
    groups = {}
    sentinel = prod(spatial)
    for linear, coordinate in enumerate(product(*(range(n) for n in out))):
        if adaptive:
            ranges = [range(c*n//o, ((c+1)*n+o-1)//o) for c,n,o in zip(coordinate,spatial,out)]
        else:
            ranges = [range(c*s-p, c*s-p+k) for c,s,p,k in zip(coordinate,stride,padding,kernel)]
        indices, valid, included = [], 0, 0
        for point in product(*ranges):
            real = all(0 <= c < n for c,n in zip(point,spatial))
            valid += real
            included += real if adaptive else all(-p <= c < n+p for c,n,p in zip(point,spatial,padding))
            index = 0
            for c,n in zip(point,spatial):
                index = index*n+c
            indices.append(index if real else sentinel)
        divisor = (params.get('divisor_override') if rank > 1 else None)
        if divisor is None:
            divisor = included if not adaptive and params.get('count_include_pad', True) else valid
        if divisor <= 0:
            raise ValueError('Empty or nonpositive pooling divisor')
        group = groups.setdefault((len(indices), divisor), {'positions': [], 'indices': []})
        group['positions'].append(linear)
        group['indices'].append(indices)
    return tuple(shape), tuple(output), groups


def average_pool(case):
    try:
        shape, output_shape, groups = pool_plan(case._call.signature.name, case.params)
    except ValueError as exc:
        _unsupported(str(exc))
    x, = case.inputs
    _bf16(x)
    _device(x)
    _, functions, hashes = load_native('bf16_gemm_nt')
    native = functions['bf16_gemm_nt']

    def run(x):
        import torch
        import torch.nn.functional as F
        flat = F.pad(x.reshape(shape[0]*shape[1], -1), (0, 1))
        output = torch.empty((flat.shape[0], prod(output_shape[2:])), dtype=x.dtype, device=x.device)
        for (width, divisor), group in groups.items():
            indices = torch.tensor(group['indices'], dtype=torch.int64, device=x.device)
            positions = torch.tensor(group['positions'], dtype=torch.int64, device=x.device)
            rows = flat.index_select(1, indices.reshape(-1)).reshape(-1, width)
            ones = torch.ones((1, width), dtype=x.dtype, device=x.device)
            lhs, rhs = _nt_operands(rows, ones)
            values = torch.empty((rows.shape[0], 1), dtype=x.dtype, device=x.device)
            native(lhs, rhs, values, alpha=1.0/divisor)
            output.index_copy_(1, positions, values.reshape(flat.shape[0], -1))
        return output.reshape(output_shape)

    return candidate(run, 'deep_gemm.bf16_gemm_nt(alpha=1/original_pool_divisor)',
                     'Gather complete original windows, zero-pad out-of-input locations, group equal original divisors, unchanged native contraction/Alpha, restore every NC-spatial output. No split K or input arithmetic.', hashes)


def mean_pooling(case):
    p = case.params
    shape, dtype = p['x']
    chunk = p['chunk_size']
    if dtype != 'bfloat16' or p['output'][1] != dtype or p['accum_dtype'] != 'float32':
        _unsupported('Uniform BF16 inputs and original FP32 accumulation required')
    if 'offsets' in p or 'indices' in p or len(shape) != 4 or shape[1] % chunk:
        _unsupported('Only complete uniform original chunks are admitted')
    output_shape = (shape[0], shape[1]//chunk, shape[2], shape[3])
    if tuple(p['output'][0]) != output_shape:
        _unsupported('Original mean-pooling output geometry differs')
    _bf16(*case.inputs)
    _device(*case.inputs)
    _, functions, hashes = load_native('bf16_gemm_nt')
    native = functions['bf16_gemm_nt']

    def run(x, offsets=None, indices=None):
        import torch
        if offsets is not None or indices is not None:
            _unsupported('Uniform chunk route cannot ignore supplied metadata')
        rows = x.reshape(shape[0], shape[1]//chunk, chunk, shape[2], shape[3]).permute(0,1,3,4,2).contiguous().reshape(-1, chunk)
        ones = torch.ones((1, chunk), dtype=x.dtype, device=x.device)
        lhs, rhs = _nt_operands(rows, ones)
        output = torch.empty((rows.shape[0], 1), dtype=x.dtype, device=x.device)
        native(lhs, rhs, output, alpha=1.0/chunk)
        return output.reshape(output_shape)

    return candidate(run, 'deep_gemm.bf16_gemm_nt(alpha=1/original_chunk_size)',
                     'Reshape/permute each complete uniform sequence chunk into K; unchanged BF16 GEMM/FP32 accumulation and Alpha; return original full output.', hashes)
