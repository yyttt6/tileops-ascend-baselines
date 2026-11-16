import math
import torch
import torch.nn.functional as F
def build_sub(case,fn):
    x, y = case.inputs
    shape = torch.broadcast_shapes(x.shape, y.shape)
    size = math.prod(shape)
    tail = -size % 4096
    alpha = case.params.get('alpha', 1)
    def pack(t):
        value = t.expand(shape).contiguous().reshape(-1)
        return (F.pad(value, (0, tail)) if tail else value).reshape(-1, 4096)
    def run(x, y):
        return fn(pack(y), pack(x), -alpha).reshape(-1)[:size].reshape(shape)
    return run
