def build(fn, output_dtype):
    def run(q, scale):
        import torch
        if scale.numel()!=1 or q.ndim<1:
            raise ValueError('per-tensor scalar scale only')
        xx=q.float().contiguous().reshape(-1,1,q.shape[-1])
        ss=scale.float().reshape(1).expand(q.shape[-1]).contiguous()
        zeros=torch.zeros_like(xx)
        return fn(xx,ss,zeros,scale_constant=0.0).reshape(q.shape).to(output_dtype)
    return run
