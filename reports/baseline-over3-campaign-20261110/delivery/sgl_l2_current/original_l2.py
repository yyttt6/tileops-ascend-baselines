import torch,math
from row_layout_original import row_layout
def build_affine(fn):
    def run(x, y, factor):
        import torch
        import torch.nn.functional as F
        shape = torch.broadcast_shapes(x.shape, y.shape)
        xx = x.expand(shape).contiguous().reshape(-1)
        yy = y.expand(shape).contiguous().reshape(-1)
        size = xx.numel()
        tail = -size % 4096
        return fn(F.pad(xx, (0, tail)).reshape(-1, 4096), F.pad(yy, (0, tail)).reshape(-1, 4096), factor).reshape(-1)[:size].reshape(shape)
    return run
def build_reciprocal(norm):
    def inverse_positive(x):
        rows = x.float().contiguous().reshape(-1, 1)
        return norm(rows, torch.ones(1, device=x.device, dtype=torch.float32), None, 0.0, is_rms_norm=True)[2].reshape(x.shape)
    return inverse_positive
def build(case,norm,reciprocal,affine):
    pack,width,shape=row_layout(case)
    x=case.inputs[0]
    assert width*x.element_size()<=65536, "original API feature limit"
    dtype=getattr(torch,case.params.get("dtype") or str(x.dtype).split(".")[-1])
    factor=math.sqrt(width)
    def run(x):
        rows = pack(x)
        rstd = norm(rows, torch.ones(width, device=x.device, dtype=x.dtype), None, 0.0, is_rms_norm=True)[2]
        rms = reciprocal(rstd)
        return affine(rms, torch.zeros_like(rms), factor).to(dtype).reshape(shape)
    return run
