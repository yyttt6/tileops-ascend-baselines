"""Full FP8 block dot, then original FP32 scales and FP32 reduction."""
import torch
import torch_npu

def factory(params):
    m,k=params['a'][0];n=params['b'][0][0];assert k%128==0
    groups=k//128;out_dtype=getattr(torch,params['out_dtype'])
    assert params['scale_a'][0]==[m,groups] and params['scale_b'][0]==[n,groups]
    unit=torch.ones(1,dtype=torch.float32,device='npu')
    def run(a,b,scale_a,scale_b,bias=None):
        aa=a.reshape(m,groups,128).permute(1,0,2).contiguous()
        bb=b.reshape(n,groups,128).permute(1,0,2).contiguous().transpose(1,2)
        partial=torch_npu.npu_quant_matmul(aa,bb,unit,output_dtype=torch.float32)
        assert partial.dtype==torch.float32 and tuple(partial.shape)==(groups,m,n)
        out=(partial*scale_a.T.contiguous()[:,:,None]*scale_b.T.contiguous()[:,None,:]).sum(0,dtype=torch.float32)
        if bias is not None:out=out+bias.float()
        return out.to(out_dtype)
    return run
