"""Original AttentionUpdate performs log-domain reductions; no external exp/log."""
from pathlib import Path

def log_reduction(case):
    import math,torch
    from dataclasses import replace
    from .native import GraphCall
    from .adapters import _candidate
    from .routing_arithmetic import weighted_rows
    from benchmarks.ascend_baselines import UnsupportedCase,WORKSPACE,source_hashes
    from .hashing import source_hashes
    x=case.inputs[0];p=case.params;softmax=case._call.signature.name=='LogSoftmaxFwdOp'
    if x.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise UnsupportedCase('Original log-domain API requires exact floating-to-FP32 widening.')
    ds=p['dim'];ds=(ds,) if isinstance(ds,int) else ds
    dims=sorted(d%x.ndim for d in ds);kept=[i for i in range(x.ndim) if i not in dims]
    binding=GraphCall('ops-transformer')
    def run(x):
        width=math.prod(x.shape[d] for d in dims)
        packed=x.permute(kept+dims).contiguous().float().reshape(-1,width);values=packed
        while values.shape[1]>1:
            count=values.shape[1];sp=min(16,count);blocks=count//sp;take=blocks*sp
            grouped=values[:,:take].reshape(values.shape[0],blocks,sp)
            log_inputs=[grouped[:,:,i].contiguous().flatten() for i in range(sp)]
            zero=torch.zeros((values.shape[0]*blocks,8),dtype=torch.float16,device=x.device)
            out=torch.empty_like(zero);lse=torch.empty_like(log_inputs[0])
            binding.call('AttentionUpdate',[*log_inputs,*([zero]*sp)],[out,lse],{'update_type':1,'sp':sp},names=[*[f'lse{i}' for i in range(sp)],*[f'go{i}' for i in range(sp)],'output','lse_m'])
            reduced=lse.reshape(values.shape[0],blocks)
            values=torch.cat((reduced,values[:,take:]),1) if take<count else reduced
        if softmax:
            left=packed.flatten();right=values.expand_as(packed).flatten()
            pairs=torch.stack((left,right),1);coeff=torch.tensor([1.,-1.],dtype=torch.float32,device=x.device)[:,None]
            result=weighted_rows(binding,pairs.reshape(-1,2,1),coeff.flatten()[None].expand(len(pairs),2).contiguous()).reshape(*[x.shape[i] for i in kept],width)
            result=result.movedim(-1,dims[0])
            return result.to(getattr(torch,p['dtype']) if p.get('dtype') else x.dtype)
        shape=[1 if i in dims else size for i,size in enumerate(x.shape)] if p['keepdim'] else [x.shape[i] for i in kept]
        return values.reshape(shape).to(x.dtype)
    result=_candidate(binding,run,'AttentionUpdate'+('+MoeFinalizeRoutingV2' if softmax else ''),'attention/attention_update','Treat each input as a log-domain singleton state with zero value; original native AttentionUpdate merges up to 16 states per call and returns complete natural LSE. Carry unpaired columns by layout copies. LogSoftmax additionally uses original FP32 routing weighted sum with [1,-1] coefficients. All packing/casts/zero buffers included; no external exp/log/softmax arithmetic.')
    return replace(result,source_sha256=result.source_sha256|source_hashes(Path(__file__),Path(__file__).with_name('routing_arithmetic.py'),WORKSPACE/'ascend_baselines/ops-transformer/moe/moe_finalize_routing_v2'))
