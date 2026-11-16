from math import prod
from pathlib import Path
from .runtime import validate,module,TRITON_ROOT,candidate
def sum_original(case):
    validate(case)
    import torch
    mean, sources = module(TRITON_ROOT + 'batch_invariant/mean.py')
    scale, extra = module(TRITON_ROOT + 'muls_add.py', hardware=True)
    p = case.params
    rank = len(p['x'][0])
    dims = p.get('dim')
    dims = tuple(range(rank)) if dims is None or dims == [] else (dims,) if isinstance(dims, int) else tuple(dims)
    dims = tuple((d % rank for d in dims))
    order = tuple((d for d in range(rank) if d not in dims)) + dims

    def run(x):
        width = prod((x.shape[d] for d in dims))
        packed = x.permute(order).contiguous().reshape(-1, width).float()
        avg = mean.mean_dim(packed, 1, dtype=torch.float32).reshape(-1, 1)
        result = scale.muls_add_triton(avg, torch.zeros_like(avg), float(width))
        return result.to(getattr(torch, p['output'][1])).reshape(p['output'][0])
    return candidate(run, 'mean_dim + muls_add_triton(sum specialization)', 'Flatten complete reduction axes, native FP32 mean followed by native scalar multiplication by the domain size; restore manifest dtype/keepdim shape. No Torch reduction. Extra rounding remains subject to the unchanged numerical oracle.', [Path(__file__), *sources, *extra])
def sum_fixed(case):
    validate(case)
    import torch
    mean, sources = module(TRITON_ROOT + 'batch_invariant/mean.py')
    scale, extra = module(TRITON_ROOT + 'muls_add.py', hardware=True)
    p = case.params
    rank = len(p['x'][0])
    dims = p.get('dim')
    dims = tuple(range(rank)) if dims is None or dims == [] else (dims,) if isinstance(dims, int) else tuple(dims)
    dims = tuple((d % rank for d in dims))
    order = tuple((d for d in range(rank) if d not in dims)) + dims

    def run(x):
        width = prod((x.shape[d] for d in dims))
        if len(dims) == 1:
            avg = mean.mean_dim(x.float(), dims[0], dtype=torch.float32).reshape(-1, 1)
        else:
            packed = x.permute(order).contiguous().reshape(-1, width).float()
            avg = mean.mean_dim(packed, 1, dtype=torch.float32).reshape(-1, 1)
        result = scale.muls_add_triton(avg, torch.zeros_like(avg), float(width))
        return result.to(getattr(torch, p['output'][1])).reshape(p['output'][0])
    return candidate(run, 'mean_dim + muls_add_triton(sum specialization)', 'Flatten complete reduction axes, native FP32 mean followed by native scalar multiplication by the domain size; restore manifest dtype/keepdim shape. No Torch reduction. Extra rounding remains subject to the unchanged numerical oracle.', [Path(__file__), *sources, *extra])
