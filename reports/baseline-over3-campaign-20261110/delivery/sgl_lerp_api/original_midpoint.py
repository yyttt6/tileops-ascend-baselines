import torch
def build(fn):
    def run(x, y):
        shape = torch.broadcast_shapes(x.shape, y.shape)
        pairs = torch.stack((x.expand(shape), y.expand(shape)), dim=-1).reshape(-1, 2)
        return fn(pairs, torch.ones(2, device=x.device, dtype=x.dtype), None, 1e-05)[1].to(x.dtype).reshape(shape)
    return run
