"""Exact native INT32 contraction with the original ordered FP32 epilogue."""
import torch
import torch_npu

def factory(params):
    m,k=params['a'][0];n,kb=params['b'][0]
    assert k==kb and k*128*128 < 2**31
    assert params['a'][1]==params['b'][1]=='int8'
    out_dtype=getattr(torch,params['out_dtype'])
    unit=torch.ones(1,device='npu',dtype=torch.float32)
    def run(a,b,scale_a,scale_b,bias=None):
        # INT32 output requires no per-token multiplier; all real scales follow
        # the exact contraction in the same FP32 order as the frozen reference.
        acc=torch_npu.npu_quant_matmul(a,b.T,unit,output_dtype=torch.int32)
        assert acc.dtype==torch.int32
        out=acc.float()*scale_a.float()*scale_b.float()
        if bias is not None:out=out+bias.float()
        return out.to(out_dtype)
    return run
