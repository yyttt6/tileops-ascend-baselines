"""Per-query legal context crops through the unchanged noncausal attention API."""
from pathlib import Path
from types import SimpleNamespace
from .attention import _attention


def _native(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
    p=case.params
    proxy=SimpleNamespace(inputs=case.inputs,params=p|{'is_causal':False,'window_size_left':-1,'window_size_right':-1})
    attend,hashes=_attention(proxy)
    hashes.update(source_hashes(Path(__file__)))
    def run(q,k,v):
        sq,sk=q.shape[0],k.shape[0]
        if sq==0:return torch.empty_like(q)
        causal=p.get('is_causal',True)
        left,right=p.get('window_size_left',-1),p.get('window_size_right',-1)
        if not causal and left<0 and right<0:return attend(q,k,v)
        out=[]
        for i in range(sq):
            position=sk-sq+i
            lo=0 if left<0 else max(0,position-left)
            hi=sk if right<0 else min(sk,position+right+1)
            if causal:hi=min(hi,position+1)
            if hi<=lo:raise UnsupportedCase('Empty query context requires a separately specified zero/LSE convention.')
            out.append(attend(q[i:i+1],k[lo:hi],v[lo:hi]))
        return torch.cat(out)
    return run,hashes


def dense(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase
    if any(t is not None for t in case.inputs[3:]):raise UnsupportedCase('Original kernel has no scaling/RoPE/sink tensor API.')
    native,hashes=_native(case)
    def run(q,k,v,*extra):return torch.stack([native(q[b],k[b],v[b]) for b in range(q.shape[0])])
    return Candidate(Implementation(run),'tilelang::flash_attention_fwd over per-query context crops',
                     'Each query selects exactly its causal/window key interval by layout slicing. Unchanged original attention handles each cropped context; equal KV repetition and original feature padding retained. Compiled shape kernels may be cached; input-derived operands are always rebuilt. Many unique context lengths may be resource-limited.',hashes)


def paged(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase
    if any(t is not None for t in case.inputs[6:]):raise UnsupportedCase('Original kernel has no scaling/RoPE/sink tensor API.')
    native,hashes=_native(case)
    def run(q,k,v,table,lens,cuq,*extra):
        lengths=lens.cpu().tolist();bounds=cuq.cpu().tolist();out=[]
        for b,length in enumerate(lengths):
            ids=table[b,:(length+k.shape[1]-1)//k.shape[1]].long()
            kk=k.index_select(0,ids).flatten(0,1)[:length];vv=v.index_select(0,ids).flatten(0,1)[:length]
            out.append(native(q[bounds[b]:bounds[b+1]],kk,vv))
        return torch.cat(out)
    return Candidate(Implementation(run),'tilelang::flash_attention_fwd over per-query gathered context crops',
                     'Read current page table/cache lengths/query offsets each call; gather pages, then slice each query to its causal/window context. Original native attention performs all score/softmax/value math; original dtype/feature packing only. Unique context compilation can be resource-limited.',hashes,timing_mode='host_roundtrip')
