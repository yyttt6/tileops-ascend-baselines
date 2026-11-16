"""Window layout adapters for original ReduceMax/ReduceMean (no new kernel math)."""
from itertools import product
from math import prod
from pathlib import Path
from types import SimpleNamespace
from .specs import files

def make(case,op):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
    from .graph import make as graph_make,deployment_hashes
    p=case.params;maximum='Max' in op;adaptive='Adaptive' in op;ndim=2 if adaptive else int(op.split('Pool')[1][0]);shape,dtype=p['input'];output_shape=p['output'][0];prefix=tuple(shape[:-ndim]);spatial=tuple(shape[-ndim:]);out_spatial=tuple(output_shape[-ndim:]);children={}
    override=p.get('divisor_override')
    def tup(v):return tuple(v) if isinstance(v,(list,tuple)) else (v,)*ndim
    if not adaptive:
        kernel=tup(p['kernel_size']);stride=tup(p.get('stride') or p['kernel_size']);pad=tup(p.get('padding',0));dilation=tup(p.get('dilation',1));effective=tuple((k-1)*d+1 for k,d in zip(kernel,dilation))
    fast=not adaptive and (maximum or (p.get('count_include_pad',True) and not p.get('ceil_mode',False)))
    groups={}
    if not fast:
        for flat,position in enumerate(product(*(range(o) for o in out_spatial))):
            if adaptive:
                starts=tuple(i*s//o for i,s,o in zip(position,spatial,out_spatial));ends=tuple(((i+1)*s+o-1)//o for i,s,o in zip(position,spatial,out_spatial));steps=(1,)*ndim
            else:
                include=p.get('count_include_pad',True);sizes=tuple(s+2*z for s,z in zip(spatial,pad)) if include else spatial
                starts0=tuple(i*t-(0 if include else z) for i,t,z in zip(position,stride,pad));starts=tuple(max(0,s) for s in starts0);ends=tuple(min(s+k,n) for s,k,n in zip(starts0,kernel,sizes));steps=(1,)*ndim
            extents=tuple(e-s for s,e in zip(starts,ends))
            if min(extents)<=0:raise UnsupportedCase('Empty pooling window in manifest.')
            groups.setdefault(extents,[]).append((flat,tuple(slice(s,e,t) for s,e,t in zip(starts,ends,steps))))
    def reduce(rows):
        # Only compiled shape/metadata are cached; every input packing is in run.
        native=rows.float() if (not maximum and rows.dtype in (torch.float16,torch.bfloat16)) else rows
        key=(tuple(native.shape),native.dtype)
        if key not in children:
            params={'output':((native.shape[0],),str(native.dtype).split('.')[-1]),'dim':-1,'keepdim':False,'_weighted':override is not None}
            children[key]=graph_make(SimpleNamespace(params=params),'AmaxFwdOp' if maximum else 'MeanFwdOp')
        args=(native,) if override is None else (native,torch.ones_like(native),torch.full_like(native,float(override)))
        return children[key].implementation.run(*args).to(rows.dtype)
    def run(x):
        if fast:
            pads=[]
            for size,k,t,z,o in reversed(list(zip(spatial,effective,stride,pad,out_spatial))):pads.extend((z,max(z,(o-1)*t+k-size-z)))
            data=F.pad(x,tuple(pads),value=float('-inf') if maximum else 0.) if any(pads) else x
            for i,(k,t) in enumerate(zip(effective,stride)):data=data.unfold(len(prefix)+i,k,t)
            data=data[tuple(slice(None) for _ in prefix)+tuple(slice(0,o) for o in out_spatial)+tuple(slice(None,None,d) for d in dilation)]
            rows=data.contiguous().reshape(-1,prod(kernel));return reduce(rows).reshape(output_shape)
        data=x
        if not adaptive and p.get('count_include_pad',True):
            pads=tuple(v for z in reversed(pad) for v in (z,z));data=F.pad(x,pads,value=0.)
        output=torch.empty((*prefix,prod(out_spatial)),device=x.device,dtype=x.dtype)
        for extents,parts in groups.items():
            packed=torch.stack([data[(slice(None),)*len(prefix)+slices].reshape(*prefix,-1) for _,slices in parts],dim=-2)
            value=reduce(packed.reshape(-1,prod(extents))).reshape(*prefix,len(parts))
            indices=torch.tensor([i for i,_ in parts],device=x.device,dtype=torch.int64)
            output.index_copy_(-1,indices,value)
        return output.reshape(output_shape)
    def cleanup():
        for c in children.values():c.cleanup()
    weighted_sources=[q for part in ('op_host','op_kernel','op_graph') for q in (Path('/root/dyq/workspace/ascend_baselines/ops-math/math/reduce_mean_with_count')/part).rglob('*') if q.is_file() and q.suffix in ('.cpp','.h')] if override is not None else []
    return Candidate(Implementation(run),'ops-math::'+('ReduceMax' if maximum else 'ReduceMean')+' (window packing)',
      'Per-call window view/packing, original stride/dilation/padding/ceil/adaptive bounds; reduce only declared divisor region. Custom divisor selects original ReduceMeanWithCount with constant weights/divisors. Half mean widens exactly to FP32 and casts output back.',
      source_hashes(Path(__file__),Path(__file__).with_name('graph.py'),*files('AmaxFwdOp' if maximum else 'MeanFwdOp'),*weighted_sources)|deployment_hashes(),cleanup=cleanup)
