"""Pooling as input windows for original GEMM/reducer APIs; no new kernel body."""
from collections import defaultdict
from itertools import product
from math import prod
from pathlib import Path
from .adapters import _gemm_native
from .microblocks import _load_builder


def _engine(maximum):
    import torch
    import torch.nn.functional as F
    from benchmarks.ascend_baselines import LIBRARIES,source_hashes
    from benchmarks.ascend_tilelang_baselines import _runtime,_runtime_hashes
    kernels={}
    hashes=source_hashes(Path(__file__))|_runtime_hashes(_runtime())
    if maximum:
        tl,builder,more=_load_builder('testing/ascend/language/test_tilelang_ascend_reduce.py','make_default_layout_reduce_kernel')
        hashes.update(more)
    else:hashes.update(source_hashes(LIBRARIES/'tilelang/examples/ascend/example_gemm.py'))
    def reduce(data,divisor):
        data=data.float().contiguous()
        m,k=data.shape
        if not maximum:
            pieces=[]
            for start in range(0,m,4096):
                block=data[start:start+4096]
                key=block.shape
                if key not in kernels:
                    kernels[key],more=_gemm_native(block.shape[0],1,k,torch.float32);hashes.update(more)
                weights=torch.full((1,k),1/divisor,device=data.device,dtype=torch.float32)
                pieces.append(kernels[key](block,weights).flatten())
            return torch.cat(pieces)
        while True:
            width=min(128,1<<(data.shape[1]-1).bit_length())
            rows=256//width
            if width not in kernels:
                kernels[width]=tl.compile(builder(rows,width,'float32','max'),target='ascend',pass_configs={tl.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE:False})
            packed=F.pad(data,(0,-data.shape[1]%width),value=-float('inf')).reshape(-1,width)
            n=packed.shape[0]
            packed=F.pad(packed,(0,0,0,-n%rows),value=-float('inf'))
            out=torch.empty(packed.shape[0],device=data.device,dtype=torch.float32)
            for start in range(0,packed.shape[0],rows):kernels[width](packed[start:start+rows],out[start:start+rows])
            data=out[:n].reshape(m,-1)
            if data.shape[1]==1:return data.flatten()
    return reduce,hashes


def fixed(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    op=case._call.signature.name
    maximum=op.startswith('Max')
    nd=next(d for d in (1,2,3) if str(d)+'d' in op)
    x,=case.inputs;p=case.params
    shape=tuple(case._call.tensors['output'][0]);outs=shape[-nd:];spatial=x.shape[-nd:]
    prefix=x.shape[:-nd];outer=prod(prefix);positions=prod(outs)
    def dims(v):return (v,)*nd if isinstance(v,int) else tuple(v)
    ks=dims(p['kernel_size']);stride=dims(p['stride']) if p.get('stride') is not None else ks
    padding=dims(p['padding']);dilation=dims(p.get('dilation',1))
    right=tuple(max(padding[i],(outs[i]-1)*stride[i]+dilation[i]*(ks[i]-1)+1-spatial[i]-padding[i]) for i in range(nd))
    pad=tuple(v for pair in reversed(list(zip(padding,right))) for v in pair)
    groups=defaultdict(list)
    for flat,coords in enumerate(product(*(range(n) for n in outs))):
        divisor=p.get('divisor_override')
        if divisor is None:
            counts=[]
            for i,j in enumerate(coords):
                start=j*stride[i]-padding[i]
                lo=-padding[i] if p.get('count_include_pad',True) else 0
                hi=spatial[i]+padding[i] if p.get('count_include_pad',True) else spatial[i]
                counts.append(max(0,min(start+ks[i],hi)-max(start,lo)))
            divisor=prod(counts)
        groups[divisor].append(flat)
    native,hashes=_engine(maximum)
    def run(x):
        padded=F.pad(x,pad,value=-float('inf') if maximum else 0.)
        st=padded.stride()
        view=padded.as_strided((*prefix,*outs,*ks),(*st[:-nd],*(st[-nd+i]*stride[i] for i in range(nd)),*(st[-nd+i]*dilation[i] for i in range(nd))))
        windows=view.contiguous().reshape(outer,positions,prod(ks))
        out=torch.empty((outer,positions),device=x.device,dtype=torch.float32)
        if maximum:
            out.copy_(native(windows.flatten(0,1),1).reshape(outer,positions))
        else:
            for divisor,ids in groups.items():
                index=torch.tensor(ids,device=x.device,dtype=torch.int64)
                values=native(windows.index_select(1,index).reshape(-1,prod(ks)),divisor).reshape(outer,len(ids))
                out.index_copy_(1,index,values)
        return out.to(x.dtype).reshape(shape)
    return Candidate(Implementation(run),'tilelang::native max reducer over windows' if maximum else 'tilelang::native GEMM constant projection over windows',
                     'Original input windows via pad/as_strided; dilation/ceil output follows exact manifest shape. Max uses original reducer shape parameters with256 threads. Average groups positions by shape-derived divisor, excluding ceil-only padding and respecting override. All packing and copies occur in run; no Torch pool/reduction.',hashes)


def adaptive(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    x,=case.inputs
    maximum=case._call.signature.name=='AdaptiveMaxPool2dFwdOp'
    shape=tuple(case._call.tensors['output'][0]);oh,ow=shape[-2:];h,w=x.shape[-2:]
    outer=prod(x.shape[:-2]);groups=defaultdict(list)
    for i,j in product(range(oh),range(ow)):
        h0,h1=i*h//oh,((i+1)*h+oh-1)//oh
        w0,w1=j*w//ow,((j+1)*w+ow-1)//ow
        groups[(h1-h0)*(w1-w0)].append((i*ow+j,h0,h1,w0,w1))
    native,hashes=_engine(maximum)
    def run(x):
        out=torch.empty((outer,oh*ow),device=x.device,dtype=torch.float32)
        for length,entries in groups.items():
            windows=torch.stack([x[...,h0:h1,w0:w1].contiguous().reshape(outer,length) for _,h0,h1,w0,w1 in entries],dim=1)
            values=native(windows.reshape(-1,length),length).reshape(outer,len(entries))
            ids=torch.tensor([v[0] for v in entries],device=x.device,dtype=torch.int64)
            out.index_copy_(1,ids,values)
        return out.to(x.dtype).reshape(shape)
    return Candidate(Implementation(run),'tilelang::original '+('max reducer' if maximum else 'GEMM constant projection')+' over adaptive regions',
                     'Exact adaptive floor/ceil region layout, grouped only by window size; original native API performs all reductions. Handles unbatched prefix dimensions; no Torch pooling/reduction.',hashes)


def mean_chunks(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    chunk=case.params['chunk_size']
    native,hashes=_engine(False)
    def run(x,offsets=None,indices=None):
        if offsets is None:slices=[(s,min(s+chunk,x.shape[1])) for s in range(0,x.shape[1],chunk)]
        else:
            bounds=offsets.cpu().tolist()
            slices=[(bounds[s]+c*chunk,min(bounds[s]+(c+1)*chunk,bounds[s+1])) for s,c in indices.cpu().tolist()]
        groups=defaultdict(list)
        for i,(start,end) in enumerate(slices):groups[end-start].append((i,start,end))
        out=torch.empty((x.shape[0],len(slices),*x.shape[2:]),device=x.device,dtype=x.dtype)
        for length,entries in groups.items():
            packed=torch.stack([x[:,start:end].movedim(1,-1).contiguous() for _,start,end in entries],dim=1)
            values=native(packed.reshape(-1,length),length).reshape(x.shape[0],len(entries),*x.shape[2:]).to(x.dtype)
            ids=torch.tensor([v[0] for v in entries],device=x.device,dtype=torch.int64)
            out.index_copy_(1,ids,values)
        return out
    return Candidate(Implementation(run),'tilelang::native GEMM constant projection over sequence chunks',
                     'Read current offsets/indices every invocation; pack named chunks with actual tail divisor and preserve requested chunk order. Exact FP32 widening and original GEMM projection; no Torch mean.',hashes,timing_mode='host_roundtrip')
