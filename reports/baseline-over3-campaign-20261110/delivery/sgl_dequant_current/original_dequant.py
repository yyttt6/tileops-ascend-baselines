import torch
from types import SimpleNamespace
def build(fn,name,output_dtype):
    def run(x, y):
        shape = torch.broadcast_shapes(x.shape, y.shape)
        xx = x.expand(shape).contiguous().reshape(1, 1, -1)
        yy = y.expand(shape).contiguous().reshape(-1)
        zeros = torch.zeros_like(xx)
        return fn(xx, yy, zeros, scale_constant=0.0).reshape(shape)
    native=SimpleNamespace(implementation=SimpleNamespace(run=run))
    def run(q, scale):
        if name == 'INT8DequantPerChannelFwdOp':
            scale = scale[:, None]
        elif name == 'INT8DequantPerBlockFwdOp':
            scale = scale.repeat_interleave(128, dim=1)[:, :q.shape[1]]
        return native.implementation.run(q.float(), scale).to(output_dtype)
    return run
