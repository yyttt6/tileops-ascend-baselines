"""Window packing into original ArgMaxWithValue, with index-coordinate layout."""
from itertools import product
from pathlib import Path
from math import prod
from types import SimpleNamespace
from .specs import SOURCE

def make(case,op):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
    from .graph import make as graph_make,deployment_hashes
    p=case.params;adaptive=op.startswith('Adaptive');nd=2 if adaptive else int(op.split('Pool')[1][0]);shape,dtype=p['input'];out_shape=p['output'][0];prefix=tuple(shape[:-nd]);spatial=tuple(shape[-nd:]);outs=tuple(out_shape[-nd:]);groups={};children={}
    def tup(x):return tuple(x) if isinstance(x,(list,tuple)) else (x,)*nd
    if not adaptive:kernel=tup(p['kernel_size']);stride=tup(p.get('stride') or p['kernel_size']);pad=tup(p.get('padding',0));dilation=tup(p.get('dilation',1))
    for flat,pos in enumerate(product(*(range(n) for n in outs))):
        if adaptive:
            coords=[list(range(i*n//o,((i+1)*n+o-1)//o)) for i,n,o in zip(pos,spatial,outs)]
        else:
            coords=[[i*s-z+j*d for j in range(k) if 0<=i*s-z+j*d<n] for i,s,z,k,d,n in zip(pos,stride,pad,kernel,dilation,spatial)]
        if any(not c for c in coords):raise UnsupportedCase('Manifest contains an empty maximum window.')
        slices=tuple(slice(c[0],c[-1]+1,c[1]-c[0] if len(c)>1 else 1) for c in coords)
        indices=[]
        for co in product(*coords):
            idx=0
            for v,n in zip(co,spatial):idx=idx*n+v
            indices.append(idx)
        groups.setdefault(tuple(map(len,coords)),[]).append((flat,slices,indices))
    def run(x):
        values=torch.empty((*prefix,prod(outs)),dtype=x.dtype,device=x.device);indices=torch.empty_like(values,dtype=torch.int64)
        for extents,windows in groups.items():
            packed=torch.stack([x[(slice(None),)*len(prefix)+s].reshape(*prefix,-1) for _,s,_ in windows],dim=-2)
            rows=packed.reshape(-1,prod(extents));key=tuple(rows.shape)
            if key not in children:children[key]=graph_make(SimpleNamespace(params={'output':((rows.shape[0],),dtype),'_argvalue':True}),'AmaxFwdOp')
            local,value=children[key].implementation.run(rows)
            local=local.reshape(*prefix,len(windows));value=value.reshape_as(local)
            # The lookup is fixed shape metadata; its values are original input coordinates.
            table=torch.tensor([ids for _,_,ids in windows],device=x.device,dtype=torch.int64)
            table=table.reshape((1,)*len(prefix)+table.shape).expand(*prefix,*table.shape)
            mapped=table.gather(-1,local.unsqueeze(-1)).squeeze(-1)
            target=torch.tensor([i for i,_,_ in windows],device=x.device,dtype=torch.int64)
            values.index_copy_(-1,target,value);indices.index_copy_(-1,target,mapped)
        return values.reshape(out_shape),indices.reshape(out_shape)
    def cleanup():
        for child in children.values():child.cleanup()
    src=[q for part in ('op_host','op_kernel','op_graph') for q in (SOURCE/'math/arg_max_with_value'/part).rglob('*') if q.is_file() and q.suffix in ('.cpp','.h','.py')]
    return Candidate(Implementation(run),'ops-math::ArgMaxWithValue (pool window layout)',
      'Pack only valid original window coordinates, preserving dilation/ceil/adaptive boundaries and first-maximum ties; native two-output max/argmax, then coordinate-table gather. All runtime input packing and lookup is timed.',
      source_hashes(Path(__file__),Path(__file__).with_name('graph.py'),*src)|deployment_hashes(),cleanup=cleanup)
