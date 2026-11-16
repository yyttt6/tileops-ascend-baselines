"""Extract original RecurrentKda's FP32 beta sigmoid from a one-hot update."""
from pathlib import Path


def sigmoid(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .providers import _native, _triton_module
    native,hashes=_native('aclnnRecurrentKda')
    tanh=case._call.signature.name=='TanhFwdOp'
    if tanh:
        conv,source=_triton_module('causal_conv1d')
        hashes.update(source_hashes(source))

    def affine(value,weight,bias=None):
        count=value.numel();extent=count+-count%256
        data=F.pad(value,(0,extent-count)).reshape(1,1,extent)
        weights=torch.full((1,extent),weight,device=value.device,dtype=torch.float32)
        bp=None if bias is None else torch.full((extent,),bias,device=value.device,dtype=torch.float32)
        out,_=conv.causal_conv1d_fwd_impl_old(data,weights,bias=bp,residual=None)
        return out.reshape(-1)[:count]

    def run(x):
        data=x.contiguous().reshape(-1)
        parts=[]
        for first in range(0,data.numel(),65536):
            value=data[first:first+65536].float()
            if tanh:value=affine(value,2.0)
            heads=value.numel()
            key=torch.zeros((1,1,heads,128),device=x.device,dtype=torch.bfloat16)
            key[...,0]=1.0
            query=torch.zeros_like(key)
            gate=torch.zeros((1,1,heads,128),device=x.device,dtype=torch.float32)
            _,state=native.npu_recurrent_kda(query,key,key,gate,value.reshape(1,1,heads),
                layout='BSND',scale=1.0,output_final_state=True,inplace_final_state=False,
                state_v_first=False,use_beta_sigmoid_in_kernel=True)
            # H0=0, g=0, k=v=e0: H1[0,0]=sigmoid(beta). The original
            # arch35 update remains FP32 through its final-state output.
            result=state[0,:,0,0].contiguous()
            if tanh:result=affine(result,2.0,-1.0)
            parts.append(result.to(x.dtype))
        if native.BACKENDS.get('npu_recurrent_kda')!='ctypes':raise RuntimeError('Wrong original recurrent backend')
        return torch.cat(parts).reshape(x.shape)

    return Candidate(Implementation(run),'FLA npu_recurrent_kda(beta sigmoid, zero state, one-hot k/v)',
        'Independent inputs become headwise FP32 beta. Original one-token native state update exposes sigmoid(beta) as FP32 state[0,0]; BF16 synthetic one-hot constants are exact, input widens losslessly. Tanh composes original FP32 tap/bias APIs as 2*sigmoid(2*x)-1. Bound each head pack to65536; all allocation, packing and native calls inside run.',
        hashes|source_hashes(Path(__file__)))
