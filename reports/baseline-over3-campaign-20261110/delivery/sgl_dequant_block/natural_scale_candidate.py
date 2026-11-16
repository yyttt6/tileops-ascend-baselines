def build(fn, output_dtype):
    def run(q, scale):
        import torch
        import torch.nn.functional as F
        if q.ndim!=2 or scale.shape!=(q.shape[0],(q.shape[1]+127)//128):
            raise ValueError('per128 block scale matrix required')
        m,k=q.shape;tail=-k%128
        xx=q.float()
        if tail:xx=F.pad(xx,(0,tail))
        xx=xx.reshape(-1,128).transpose(0,1).contiguous().reshape(128,1,-1)
        ss=scale.float().contiguous().reshape(-1)
        zeros=torch.zeros_like(xx)
        out=fn(xx,ss,zeros,scale_constant=0.0)
        return out.reshape(128,-1).transpose(0,1).contiguous().reshape(m,k+tail)[:,:k].contiguous().to(output_dtype)
    return run
