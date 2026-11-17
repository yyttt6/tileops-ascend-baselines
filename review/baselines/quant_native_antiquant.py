"""Complete torch_npu public APIs, separately attributed from eager/ops-math."""
import torch
import torch.nn.functional as F

def factory(op,params):
    import torch_npu
    if op.startswith('INT8Dequant'):
        dt=getattr(torch,params['out_dtype'])
        if dt==torch.float32:return None
        def run(q,s):
            if op=='INT8DequantPerBlockFwdOp':
                m,k=q.shape;pad=(-k)%128
                padded=F.pad(q,(0,pad)) if pad else q
                y=torch_npu.npu_anti_quant(padded.reshape(-1,128),s.reshape(-1,1),offset=None,dst_dtype=dt)
                return y.reshape(m,-1)[:,:k]
            scale=s[:,None] if op=='INT8DequantPerChannelFwdOp' else s
            return torch_npu.npu_anti_quant(q,scale,offset=None,dst_dtype=dt)
        return run
    return None
