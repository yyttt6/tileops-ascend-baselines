"""FP32 products and fixed linear maps through original routing weighted sums."""
from pathlib import Path


def weighted_rows(binding, values, weights):
    """values[N,K,H], weights[N,K]; only address/layout construction here."""
    import torch
    n, k, h = values.shape
    packed = values.contiguous().reshape(n * k, h)
    ids = torch.arange(n * k, dtype=torch.int32, device=values.device).reshape(n, k).T.contiguous().flatten()
    out = torch.empty((n, h), dtype=values.dtype, device=values.device)
    binding.call('MoeFinalizeRoutingV2', [packed, ids, None, None, None, weights.contiguous(), None],
                 [out], {'drop_pad_mode': 0, 'k': k})
    return out


def arithmetic(case):
    import torch
    import math
    from dataclasses import replace
    from .adapters import _candidate
    from .native import GraphCall
    from .hashing import source_hashes
    from benchmarks.ascend_baselines import UnsupportedCase
    op, p = case._call.signature.name, case.params
    if case.inputs[0].dtype != torch.float32 and not (op == 'AddFwdOp' and case.inputs[0].dtype in (torch.float16, torch.bfloat16)):
        raise UnsupportedCase('Routing arithmetic binds exact FP32 native inputs only.')
    if p.get('alpha', 1) != 1:
        raise UnsupportedCase('Nonunit subtraction coefficients are not mapped.')
    binding = GraphCall('ops-transformer')
    def run(x, y=None):
        if op in ('SumFwdOp', 'MeanFwdOp'):
            dims = p['dim']
            dims = range(x.ndim) if dims is None else ((dims,) if isinstance(dims, int) else dims)
            dims = sorted(d % x.ndim for d in dims)
            kept = [i for i in range(x.ndim) if i not in dims]
            width = math.prod(x.shape[i] for i in dims)
            values = x.permute(kept + dims).contiguous().reshape(-1, width, 1)
            weights = torch.full(values.shape[:2], 1 / width if op == 'MeanFwdOp' else 1., dtype=x.dtype, device=x.device)
            out = weighted_rows(binding, values, weights)
            shape = [1 if i in dims else size for i, size in enumerate(x.shape)] if p['keepdim'] else [x.shape[i] for i in kept]
            dtype = getattr(torch, p['dtype']) if p.get('dtype') else x.dtype
            return out.to(dtype).reshape(shape)
        if y is None:
            values = x.contiguous().reshape(-1, 1, 1)
            weights = torch.full(values.shape[:2], -1., dtype=x.dtype, device=x.device)
            return weighted_rows(binding, values, weights).reshape_as(x)
        a, b = torch.broadcast_tensors(x, y)
        if op == 'AddFwdOp':
            # The original x1 path copies the accumulator instead of adding
            # a leading +0. This preserves -0 + -0 for the unchanged edge oracle.
            lhs, rhs = a.float().contiguous().reshape(-1, 1), b.float().contiguous().reshape(-1, 1)
            ids = torch.arange(len(lhs), dtype=torch.int32, device=x.device)
            out = torch.empty_like(lhs)
            binding.call('MoeFinalizeRoutingV2', [rhs, ids, lhs, None, None, None, None],
                         [out], {'drop_pad_mode': 0, 'k': 1})
            return out.reshape(a.shape).to(x.dtype)
        if op == 'MulFwdOp':
            values = a.contiguous().reshape(-1, 1, 1)
            weights = b.contiguous().reshape(-1, 1)
        else:
            values = torch.stack((a, b), -1).float().reshape(-1, 2, 1)
            w = p.get('weight')
            coefficients = (1 - w, w) if op == 'LerpScalarFwdOp' else ((1., 1.) if op == 'AddFwdOp' else (1., -1.))
            weights = torch.tensor(coefficients, dtype=torch.float32, device=x.device).expand(values.shape[0], -1).contiguous()
        return weighted_rows(binding, values, weights).reshape(a.shape).to(x.dtype)
    candidate = _candidate(binding, run, 'MoeFinalizeRoutingV2', 'moe/moe_finalize_routing_v2',
        'Original native FP32 routing weighted sum: one expert for products/negation; two experts with exact constant weights for subtraction/interpolation; complete reduction width for sum/mean. Add instead uses native x1 accumulator-copy plus one unscaled expert, preserving signed zero. Metadata permutation, input packing, constants, output layout/cast all inside run. No external product/sum arithmetic.')
    return replace(candidate, source_sha256=candidate.source_sha256 | source_hashes(Path(__file__)))


def dispatch(case):
    import torch
    from dataclasses import replace
    from .hashing import source_hashes
    if case.inputs[0].dtype == torch.float32 or (case._call.signature.name == 'AddFwdOp' and case.inputs[0].dtype == torch.bfloat16):
        candidate = arithmetic(case)
    else:
        op = case._call.signature.name
        if op == 'AddFwdOp':
            from . import _existing
            candidate = _existing(op)(case)
        elif op == 'MulFwdOp':
            from .multiply import multiply
            candidate = multiply(case)
        elif op in ('SumFwdOp', 'MeanFwdOp'):
            from .linear import reduction_dispatch
            candidate = reduction_dispatch(case)
        else:
            from .linear import linear
            candidate = linear(case)
    return replace(candidate, source_sha256=candidate.source_sha256 | source_hashes(Path(__file__)))
