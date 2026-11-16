"""Original TopKV2 on each valid attention-index window, with original padding."""
from pathlib import Path
from types import SimpleNamespace
from .specs import SOURCE

def make(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .graph import make as graph_make,deployment_hashes
    p=case.params;children={};out_spec=p.get('output',p.get('indices'))
    if out_spec is None:
        out_spec=next(v for k,v in p.items() if k not in ('index_score','starts','ends') and isinstance(v,(list,tuple)) and len(v)==2 and isinstance(v[1],str))
    out_shape,out_dtype=out_spec;k=out_shape[-1]
    def run(scores,starts,ends):
        b,s,n,g=scores.shape;data=scores.permute(0,1,3,2).reshape(b*s,g,n);lo=starts.reshape(-1).tolist();hi=ends.reshape(-1).tolist();groups={}
        out=torch.full((b*s,g,k),n,device=scores.device,dtype=getattr(torch,out_dtype))
        # Original-coordinate lookup is layout metadata. Build its range once
        # per call instead of materializing millions of Python integers per
        # window; selection itself remains the original native TopKV2.
        coordinate_range=torch.arange(n,device=scores.device,dtype=torch.int64)
        for row,(a,z) in enumerate(zip(lo,hi)):
            a=max(0,a);z=min(n,z)
            if z>a:groups.setdefault(z-a,[]).append((row,a,z))
        for length,parts in groups.items():
            keep=min(k,length);packed=torch.stack([data[row,:,a:z] for row,a,z in parts]);key=tuple(packed.shape)
            if key not in children:children[key]=graph_make(SimpleNamespace(params={'output':((len(parts),g,keep),str(scores.dtype).split('.')[-1]),'_topk':keep}),'MulFwdOp')
            _,local=children[key].implementation.run(packed)
            coordinates=torch.stack([coordinate_range[a:z] for _,a,z in parts])[:,None,:].expand(len(parts),g,length)
            mapped=coordinates.gather(-1,local).to(out.dtype)
            for j,(row,_,_) in enumerate(parts):out[row,:,:keep].copy_(mapped[j])
        return out.reshape(out_shape)
    def cleanup():
        for child in children.values():child.cleanup()
    src=[q for part in ('op_host','op_kernel','op_graph') for q in (SOURCE/'math/top_k_v2'/part).rglob('*') if q.is_file() and q.suffix in ('.cpp','.h','.py')]
    return Candidate(Implementation(run),'ops-math::TopKV2 (valid-window crop)',
      'Read original window metadata each call; pack equal-length valid crops; native top-k; deterministic coordinate gather and original sentinel padding. No input-dependent data cached.',
      source_hashes(Path(__file__),Path(__file__).with_name('graph.py'),*src)|deployment_hashes(),timing_mode='host_roundtrip',cleanup=cleanup)
