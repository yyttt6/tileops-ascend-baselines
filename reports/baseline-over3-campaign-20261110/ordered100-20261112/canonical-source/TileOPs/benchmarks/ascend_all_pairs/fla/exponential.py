"""Native FLA gk exponential with identity initial state and zero updates."""
from pathlib import Path


def exponential(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .providers import _native, _triton_module
    native,hashes=_native('aclnnChunkFwdH')
    subtract_one=case._call.signature.name=='Expm1FwdOp'
    if subtract_one:
        conv,source=_triton_module('causal_conv1d')
        hashes.update(source_hashes(source))
    def run(x):
        data=x.contiguous().reshape(-1)
        parts=[]
        for first in range(0,data.numel(),65536):
            stop=min(data.numel(),first+65536);count=stop-first
            extent=count+-count%128;heads=extent//128
            gate=F.pad(data[first:stop].float(),(0,extent-count)).reshape(1,heads,1,128)
            zeros=torch.zeros((1,heads,1,128),device=x.device,dtype=torch.bfloat16)
            state=torch.ones((1,heads,128,128),device=x.device,dtype=torch.float32)
            _,_,state=native.npu_chunk_fwd_h(zeros,zeros,zeros,gk=gate,initial_state=state,output_final_state=True,use_exp2=False)
            value=state[...,0].contiguous().reshape(-1)[:count]
            if subtract_one:
                width=count+-count%256
                packed=F.pad(value,(0,width-count)).reshape(1,1,width)
                out,_=conv.causal_conv1d_fwd_impl_old(packed,torch.ones((1,width),device=x.device,dtype=torch.float32),bias=torch.full((width,),-1.0,device=x.device,dtype=torch.float32), residual=None)
                value=out.reshape(-1)[:count]
            parts.append(value.to(x.dtype))
        if native.BACKENDS.get('npu_chunk_fwd_h')!='ctypes':raise RuntimeError('Wrong original ChunkFwdH backend')
        return torch.cat(parts).reshape(x.shape)
    return Candidate(Implementation(run),'FLA npu_chunk_fwd_h(gk=input,initial_state=1,k=w=u=0)',
        'One native token: each state row becomes exp(gk)*1+0. Pack independent inputs along K and heads, retain one repeated value column; bounded constant state allocation. Expm1 composes original native bias=-1. Original kernels and schedules intact; input packing/allocation inside run.',hashes|source_hashes(Path(__file__)))
