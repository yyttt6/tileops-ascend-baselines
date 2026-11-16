import math
def row_layout(case):
    x = case.inputs[0]
    dim = case.params.get('dim', -1)
    axes = tuple(range(x.ndim)) if dim is None else (dim,) if isinstance(dim, int) else tuple(dim)
    axes = tuple((d % x.ndim for d in axes))
    kept = tuple((d for d in range(x.ndim) if d not in axes))
    width = math.prod((x.shape[d] for d in axes))
    shape = tuple((1 if d in axes else n for d, n in enumerate(x.shape))) if case.params.get('keepdim', False) else tuple((x.shape[d] for d in kept))

    def pack(x):
        return x.permute(*kept + axes).contiguous().reshape(-1, width)
    return (pack, width, shape)
