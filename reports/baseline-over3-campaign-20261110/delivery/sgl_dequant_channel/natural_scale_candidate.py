def build(fn, output_dtype):
    def run(q, scale):
        import torch
        if q.ndim!=2 or scale.ndim!=1 or scale.numel()!=q.shape[0]:
            raise ValueError('per-row scale requires matrix and matching vector')
        xx=q.float().transpose(0,1).contiguous().reshape(q.shape[1],1,q.shape[0])
        ss=scale.float().contiguous()
        zeros=torch.zeros_like(xx)
        out=fn(xx,ss,zeros,scale_constant=0.0)
        return out.reshape(q.shape[1],q.shape[0]).transpose(0,1).contiguous().to(output_dtype)
    return run
