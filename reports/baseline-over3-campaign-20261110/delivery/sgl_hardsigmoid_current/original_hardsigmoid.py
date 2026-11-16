import torch
from types import SimpleNamespace
from original_sub_body import build_sub
def maximum(fn):
    def run(x, y):
        shape = torch.broadcast_shapes(x.shape, y.shape)
        pairs = torch.stack((x.expand(shape), y.expand(shape)), dim=-1).reshape(-1, 2)
        indices, _ = fn(pairs)
        return pairs.gather(1, indices[:, None]).reshape(shape)
    return run
def minimum(native,affine):
    def sub(case):
        return SimpleNamespace(implementation=SimpleNamespace(run=build_sub(case,affine)))
    def run(x, y):
        import torch
        shape = torch.broadcast_shapes(x.shape, y.shape)
        xx, yy = (x.expand(shape).contiguous(), y.expand(shape).contiguous())
        n = sub(SimpleNamespace(inputs=(torch.zeros_like(xx), xx), params={'alpha': 1}))
        nx = n.implementation.run(torch.zeros_like(xx), xx)
        ny = n.implementation.run(torch.zeros_like(yy), yy)
        selected = native.implementation.run(nx, ny)
        return n.implementation.run(torch.zeros_like(selected), selected)
    return run
def build(argmax,affine):
    high=SimpleNamespace(implementation=SimpleNamespace(run=maximum(argmax)))
    low=SimpleNamespace(implementation=SimpleNamespace(run=minimum(high,affine)))
    name="HardsigmoidFwdOp"
    def apply_affine(x, y, factor):
        import torch.nn.functional as F
        size = x.numel()
        tail = -size % 4096
        xx = F.pad(x.contiguous().reshape(-1), (0, tail)).reshape(-1, 4096)
        yy = F.pad(y.expand_as(x).contiguous().reshape(-1), (0, tail)).reshape(-1, 4096)
        return affine(xx, yy, factor).reshape(-1)[:size].reshape_as(x)
    def run(x, weight=None):
        xf = x.float()
        zero = torch.zeros((), device=x.device, dtype=torch.float32)
        if name in ('HardsigmoidFwdOp', 'HardswishFwdOp'):
            shifted = apply_affine(xf, torch.full_like(xf, 3), 1.0)
            clipped = low.implementation.run(high.implementation.run(shifted, zero), torch.full((), 6, device=x.device, dtype=torch.float32))
            if name == 'HardswishFwdOp':
                clipped = times.implementation.run(xf, clipped)
            out = apply_affine(clipped, torch.zeros_like(clipped), 1.0 / 6)
        else:
            positive = high.implementation.run(xf, zero)
            negative = low.implementation.run(xf, zero)
            if name == 'LeakyReluFwdOp':
                out = apply_affine(negative, positive, case.params['negative_slope'])
            else:
                shape = (1, -1, *[1] * (x.ndim - 2)) if weight.numel() != 1 and x.ndim > 1 else ()
                scale = weight.float().reshape(shape)
                product = times.implementation.run(negative, scale)
                out = apply_affine(product, positive, 1.0)
        return out.to(x.dtype)
    return run
