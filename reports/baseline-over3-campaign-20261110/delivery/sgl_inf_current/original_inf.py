import torch
from types import SimpleNamespace
from row_layout_original import row_layout
from original_sub_body import build_sub
def maximum(fn):
    def run(x, y):
        shape = torch.broadcast_shapes(x.shape, y.shape)
        pairs = torch.stack((x.expand(shape), y.expand(shape)), dim=-1).reshape(-1, 2)
        indices, _ = fn(pairs)
        return pairs.gather(1, indices[:, None]).reshape(shape)
    return run
def build_absolute(case,fn,affine):
    native_max=SimpleNamespace(implementation=SimpleNamespace(run=maximum(fn)))
    sub=build_sub(SimpleNamespace(inputs=(torch.zeros_like(case.inputs[0]),case.inputs[0]),params={"alpha":1}),affine)
    def negative(x):
        return sub(torch.zeros_like(x),x)
    native_neg=SimpleNamespace(implementation=SimpleNamespace(run=negative))
    name="AbsFwdOp"
    def run(x, lo=None, hi=None):
        if name == 'AbsFwdOp':
            return native_max.implementation.run(native_neg.implementation.run(x), x)
        if name == 'ReluFwdOp':
            return native_max.implementation.run(x, torch.zeros((), device=x.device, dtype=x.dtype))
        if name in ('ClampScalarFwdOp', 'HardtanhFwdOp'):
            a = case.params.get('min', case.params.get('min_val'))
            b = case.params.get('max', case.params.get('max_val'))
            lo = None if a is None else torch.tensor(a, device=x.device, dtype=x.dtype)
            hi = None if b is None else torch.tensor(b, device=x.device, dtype=x.dtype)
        out = x
        if lo is not None:
            out = native_max.implementation.run(out, lo)
        if hi is not None:
            out = native_min.implementation.run(out, hi)
        return out
    return run
def build(case,fn,affine):
    pack,width,shape=row_layout(case)
    dtype=getattr(torch,case.params.get("dtype") or str(case.inputs[0].dtype).split(".")[-1])
    absrun=build_absolute(case,fn,affine)
    absolute=SimpleNamespace(implementation=SimpleNamespace(run=absrun))
    def argmax(value):
        return fn(pack(value))[0].reshape(shape)
    native=SimpleNamespace(implementation=SimpleNamespace(run=argmax))
    def run(x):
        values = absolute.implementation.run(x)
        indices = native.implementation.run(values).reshape(-1, 1)
        return pack(values).gather(1, indices).reshape(shape).to(dtype)
    return run
