"""Map complete reduction axes to a spatial height; use original pooling math."""
from math import prod
from .common import candidate, new_binding


def factory(case):
    import torch
    p = case.params
    binding = new_binding()
    output_shape = tuple(p['output'][0])

    def run(x):
        dims = p.get('dim')
        dims = tuple(range(x.ndim)) if dims is None else (dims,) if isinstance(dims, int) else tuple(dims)
        dims = tuple(d % x.ndim for d in dims)
        outer = tuple(d for d in range(x.ndim) if d not in dims)
        rows = prod(x.shape[d] for d in outer)
        length = prod(x.shape[d] for d in dims)
        dtype = getattr(torch, p['dtype']) if p.get('dtype') else x.dtype
        # This is a complete axis permutation/reshape, with no padding, subsets,
        # duplication, synthetic random data or changed logical reduction window.
        packed = x.permute(*outer, *dims).contiguous().to(dtype).float().reshape(rows, 1, length, 1)
        output = torch.empty((rows, 1, 1, 1), device=x.device, dtype=torch.float32)
        binding.call('AdaptiveAvgPool2d', [packed], [output], {'output_size': [1, 1]})
        return output.reshape(output_shape).to(dtype)

    return candidate(binding, run, 'AdaptiveAvgPool2d',
                     'Pack complete reduction axes into spatial height with singleton width; retain every original element, output shape/dtype, explicit dtype conversion and original FP32 accumulation. Original pool kernel and automatic tiling unchanged.',
                     'pooling/adaptive_avg_pool2d')
