import math
import torch.nn.functional as F
from types import SimpleNamespace
def row_max_case(case, factory, row_multiple=1):
    x, = case.inputs
    dims = case.params['dim']
    dims = tuple(range(x.ndim)) if dims is None else (dims,) if isinstance(dims, int) else tuple(dims)
    axes = tuple((d % x.ndim for d in dims))
    kept = tuple((d for d in range(x.ndim) if d not in axes))
    width = math.prod((x.shape[d] for d in axes))
    rows = x.numel() // width
    shape = tuple((1 if d in axes else n for d, n in enumerate(x.shape))) if case.params['keepdim'] else tuple((x.shape[d] for d in kept))

    def pack(x):
        value = x.permute(*kept + axes).contiguous().reshape(rows, width)
        return F.pad(value, (0, 0, 0, -rows % row_multiple)) if rows % row_multiple else value
    native = factory(SimpleNamespace(inputs=(pack(x),), params=case.params | {'dim': -1, 'keepdim': False}))

    def run(x):
        return native.implementation.run(pack(x)).reshape(-1)[:rows].reshape(shape)
    return run
def native_factory(case,fn):
    keepdim=case.params.get("keepdim",False)
    def run(x):
        indices, _ = fn(x.contiguous())
        values = x.gather(1, indices.unsqueeze(1))
        return values if keepdim else values.squeeze(1)
    return SimpleNamespace(implementation=SimpleNamespace(run=run))
