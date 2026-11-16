import torch
def build(fn,width,eps):
    def run(x, weight, bias):
        w = torch.ones(width, device=x.device, dtype=x.dtype) if weight is None else weight.contiguous().reshape(width)
        b = None if bias is None else bias.contiguous().reshape(width)
        return fn(x.contiguous().reshape(-1, width), w, b, eps)[0].reshape(x.shape)
    return run
