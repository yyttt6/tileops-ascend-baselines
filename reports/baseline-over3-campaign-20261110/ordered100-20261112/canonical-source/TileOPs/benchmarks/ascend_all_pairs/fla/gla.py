"""GLA from the original ChunkFwdH gk route with W=0 and native readout."""
from pathlib import Path


def gla(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    from .providers import _native, _triton_module
    if case.inputs[0].dtype!=torch.bfloat16:
        raise UnsupportedCase('Original ChunkFwdH gk state update supports BF16 only; FP16 conversion would be lossy.')
    native,hashes=_native('aclnnChunkFwdH')
    conv,source=_triton_module('causal_conv1d')
    recurrent=case._call.signature.name=='GLARecurrentFwdOp'
    p=case.params
    def readout(q,state,scale):
        # State [B,H,K,V] and q [B,H,K] become independent output channels.
        b,h,k,v=state.shape;channels=b*h*v;extent=channels+-channels%256
        values=state.permute(2,0,1,3).contiguous().reshape(k,channels)
        taps=q.permute(2,0,1).unsqueeze(-1).expand(k,b,h,v).contiguous().reshape(k,channels).float()
        accumulator=None
        for first in range(0,k,64):
            stop=min(k,first+64);count=stop-first;steps=1<<(count-1).bit_length()
            x=F.pad(values[first:stop],(0,extent-channels,steps-count,0)).unsqueeze(0)
            w=F.pad(taps[first:stop],(0,extent-channels,steps-count,0))
            residual=None
            if accumulator is not None:
                residual=torch.zeros_like(x);residual[:,-1,:]=accumulator
            out,_=conv.causal_conv1d_fwd_impl_old(x,w,residual=residual, bias=None)
            accumulator=out[:,-1,:].contiguous()
        out,_=conv.causal_conv1d_fwd_impl_old(accumulator.unsqueeze(1),torch.full((1,extent),scale,device=q.device,dtype=torch.float32), bias=None, residual=None)
        return out.reshape(-1)[:channels].reshape(b,h,v).to(q.dtype)

    def run(q,k,v,g,initial=None,offsets=None,offsets_cpu=None):
        if recurrent:q,k,v,g=(x.unsqueeze(1) for x in (q,k,v,g))
        dk,dv=q.shape[-1],v.shape[-1]
        scale=p.get('scale');scale=dk**-0.5 if scale is None or scale<=0 else scale
        metadata=offsets_cpu if offsets_cpu is not None else offsets
        spans=[(0,q.shape[1])] if metadata is None else list(zip(metadata.cpu().tolist(),metadata.cpu().tolist()[1:]))
        outputs=[];states=[]
        for sequence,(begin,end) in enumerate(spans):
            state=None if initial is None else (initial if metadata is None else initial[sequence:sequence+1])
            state=torch.zeros((q.shape[0],v.shape[2],128,128),device=q.device,dtype=torch.float32) if state is None else F.pad(state.float(),(0,128-dv,0,128-dk)).clone()
            result=[]
            for t in range(begin,end):
                key=F.pad(k[:,t:t+1],(0,128-dk)).transpose(1,2).contiguous()
                value=F.pad(v[:,t:t+1],(0,128-dv)).transpose(1,2).contiguous()
                gate=F.pad(g[:,t:t+1],(0,128-dk)).transpose(1,2).contiguous().float()
                _,_,state=native.npu_chunk_fwd_h(key,torch.zeros_like(key),value,gk=gate,initial_state=state,output_final_state=True,use_exp2=False)
                result.append(readout(q[:,t],state[...,:dk,:dv].contiguous(),scale))
            outputs.append(torch.stack(result,dim=1));states.append(state[...,:dk,:dv].contiguous())
        out=torch.cat(outputs,dim=1);final=torch.cat(states,dim=0)
        if native.BACKENDS.get('npu_chunk_fwd_h')!='ctypes':raise RuntimeError('Wrong original ChunkFwdH backend')
        return (out[:,0],final.to(q.dtype)) if recurrent else (out,final)
    return Candidate(Implementation(run),'FLA npu_chunk_fwd_h(gk,W=0,T=1) + causal_conv1d_fwd_impl_old(readout)',
        'Original one-token gk state recurrence with zero projection W removes delta correction. At T=1 prepared kg equals raw k. Keep native FP32 state; native causal FP32 tap/residual computes q-state readout and scale. Packed sequences independent, per-token work inside run, original kernels/schedules unchanged.',
        hashes|source_hashes(Path(__file__),source))
