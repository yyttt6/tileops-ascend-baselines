from math import prod
from pathlib import Path
from .runtime import validate,module,TRITON_ROOT,candidate
def mean_original(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + 'batch_invariant/mean.py')
    p = case.params
    shape = tuple(p['output'][0])
    dims = p.get('dim')
    rank = len(p['x'][0])
    dims = tuple(range(rank)) if dims is None or dims == [] else dims if isinstance(dims, (tuple, list)) else (dims,)
    dims = tuple((d % rank for d in dims))
    order = tuple((d for d in range(rank) if d not in dims)) + dims
    dtype = p.get('dtype') or p['output'][1]

    def run(x):
        width = prod((x.shape[d] for d in dims))
        packed = x.permute(order).contiguous().reshape(-1, width)
        result = native.mean_dim(packed.float(), 1, dtype=torch.float32)
        return result.to(getattr(torch, dtype)).reshape(shape)
    return candidate(run, 'mean_dim', 'Move/flatten all reduction axes to one axis; exactly widen FP16/BF16 to FP32 for native accumulation, cast original output dtype and restore keepdim shape; no torch reduction.', [Path(__file__), *sources])
def mean_fixed(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + 'batch_invariant/mean.py')
    p = case.params
    shape = tuple(p['output'][0])
    dims = p.get('dim')
    rank = len(p['x'][0])
    dims = tuple(range(rank)) if dims is None or dims == [] else dims if isinstance(dims, (tuple, list)) else (dims,)
    dims = tuple((d % rank for d in dims))
    order = tuple((d for d in range(rank) if d not in dims)) + dims
    dtype = p.get('dtype') or p['output'][1]

    def run(x):
        width = prod((x.shape[d] for d in dims))
        packed = x.permute(order).contiguous().reshape(-1, width)
        result = native.mean_dim(packed.float(), 1, dtype=torch.float32)
        return result.to(getattr(torch, dtype)).reshape(shape)
    return candidate(run, 'mean_dim', 'Move/flatten all reduction axes to one axis; exactly widen FP16/BF16 to FP32 for native accumulation, cast original output dtype and restore keepdim shape; no torch reduction.', [Path(__file__), *sources])
