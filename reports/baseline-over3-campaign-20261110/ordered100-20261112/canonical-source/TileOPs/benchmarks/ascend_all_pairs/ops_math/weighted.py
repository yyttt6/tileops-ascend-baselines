"""Layout-only selection of original weighted reduction for dot-product contracts.

Original DAG: ReduceSum(x * count / count_sum, axes). Selecting count=B and
count_sum=1 gives a dot product. Output tiling bounds temporary packing memory;
no native tiling/schedule or mathematical instruction is modified.
"""
from pathlib import Path
from types import SimpleNamespace
from math import prod
from .specs import SOURCE

def make(case,op):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
    from .graph import make as graph_make,deployment_hashes
    p=case.params;children={};calls=0
    layout=p.get('layout')
    if op=='MoEGroupedGemmFwdOp' and isinstance(layout,dict):
        # Case.params serializes the manifest layout sum type, while the
        # workload's internal call keeps its typed object. Decode metadata
        # only; all expert multiplication still uses the native reduction.
        kind,values=next(iter(layout.items()))
        if kind not in ('masked','contiguous'):raise UnsupportedCase('Unrecognized original expert layout.')
        layout=SimpleNamespace(kind=kind,**values)
    if op=='MoEGroupedGemmFwdOp' and p.get('activation') is not None:
        raise UnsupportedCase('Original weighted reduction has no fused gated activation epilogue.')
    def mm(a,b,bias=None):
        nonlocal calls
        m,k=a.shape;n=b.shape[1]
        if k<=0:raise UnsupportedCase('The original weighted-reduction kernel requires a nonempty reduction axis.')
        out=torch.empty((m,n),dtype=a.dtype,device=a.device)
        for i in range(0,m,64):
            for j in range(0,n,128):
                left=a[i:i+64];right=b[:,j:j+128].T
                if bias is not None:
                    left=torch.cat((left,torch.ones((left.shape[0],1),device=a.device,dtype=a.dtype)),dim=1)
                    right=torch.cat((right,bias[j:j+128,None]),dim=1)
                aa=left[:,None,:].expand(-1,right.shape[0],-1).float().contiguous()
                bb=right[None,:,:].expand(left.shape[0],-1,-1).float().contiguous()
                key=tuple(aa.shape)
                if key not in children:
                    children[key]=graph_make(SimpleNamespace(params={'output':(aa.shape[:-1],'float32'),'dim':-1,'keepdim':False,'_weighted':True}),'MeanFwdOp')
                value=children[key].implementation.run(aa,bb,torch.ones_like(aa))
                out[i:i+left.shape[0],j:j+right.shape[0]].copy_(value.to(a.dtype))
                calls+=1
                # Keep native descriptors and temporary tensors alive until the
                # stream drains, with bounded memory for large tiled matrices.
                if calls%8==0:
                    for child in children.values():child.cleanup()
        return out
    def run(*args):
        if op=='MHCPostFwdOp':
            x,h,res=args;b,n=h.shape;k=x.shape[-1]
            aa=torch.stack((x[:,None,:].expand(b,n,k).float(),res.reshape(b,n,k).float()),dim=-1)
            bb=torch.stack((h[:,:,None].expand(b,n,k),torch.ones((b,n,k),device=x.device,dtype=torch.float32)),dim=-1)
            key=tuple(aa.shape)
            if key not in children:children[key]=graph_make(SimpleNamespace(params={'output':(aa.shape[:-1],'float32'),'_weighted':True}),'MeanFwdOp')
            return children[key].implementation.run(aa,bb,torch.ones_like(aa)).reshape(res.shape).to(res.dtype)
        if op=='SSDChunkCouplingFwdOp':
            c,b=args;batch,total,g,state=c.shape;q=p['chunk_len'];nc=total//q
            cc=c.reshape(batch,nc,q,g,state).permute(0,1,3,2,4).reshape(-1,q,state)
            bb=b.reshape(batch,nc,q,g,state).permute(0,1,3,2,4).reshape(-1,q,state)
            values=[]
            for i in range(cc.shape[0]):
                value=mm(cc[i],bb[i].T)
                # Upper triangle is manifest-defined zero padding, not computed data.
                for row in range(q):value[row,row+1:].zero_()
                values.append(value)
            return torch.stack(values).reshape(batch,nc,g,q,q)
        if op=='MoEGroupedGemmFwdOp':
            a,b,metadata=args;meta=metadata.tolist();dtype=a.dtype if p.get('out_dtype') is None else getattr(torch,p['out_dtype'])
            out=torch.zeros((*a.shape[:-1],b.shape[1]),device=a.device,dtype=dtype)
            if layout.kind=='masked':
                for g,count in enumerate(meta):
                    if count:out[g,:count].copy_(mm(a[g,:count].float(),b[g].float().T).to(dtype))
            elif layout.metadata_kind=='per_row':
                for g in range(b.shape[0]):
                    rows=[i for i,v in enumerate(meta) if v==g]
                    if rows:
                        ids=torch.tensor(rows,device=a.device,dtype=torch.int64)
                        out.index_copy_(0,ids,mm(a.index_select(0,ids).float(),b[g].float().T).to(dtype))
            else:
                start=0
                for g,end in enumerate(meta):
                    start=-(-start//layout.alignment)*layout.alignment
                    if end>start:out[start:end].copy_(mm(a[start:end].float(),b[g].float().T).to(dtype))
                    start=end
            return out
        if op=='GemmFwdOp':
            a,b=args;return mm(a.T if p.get('trans_a') else a,b.T if p.get('trans_b') else b)
        if op=='BmmFwdOp':
            a,b=args;return torch.stack([mm(a[i],b[i]) for i in range(a.shape[0])])
        if op=='GroupedGemmFwdOp':
            a,b,sizes,offsets=args;sizes=sizes.tolist();offsets=offsets.tolist();ta=p['transpose_a'];tb=p['transpose_b'];values=[]
            for g,(size,start) in enumerate(zip(sizes,offsets)):
                end=start+size
                aa=a[start:end].T if ta else a[start:end]
                bb=(b[:,start:end].T if tb else b[start:end]) if ta else (b[g].T if tb else b[g])
                values.append(mm(aa,bb))
            return torch.stack(values) if ta else torch.cat(values,dim=0)
        if op.startswith('Conv'):
            x,w,bias=args;nd=x.ndim-2
            def tup(v):return tuple(v) if isinstance(v,(tuple,list)) else (v,)*nd
            stride=tup(p['stride']);pad=tup(p['padding']);dilation=tup(p['dilation']);kernel=tuple(w.shape[2:]);effective=tuple((k-1)*d+1 for k,d in zip(kernel,dilation));g=p['groups'];cin=x.shape[1]//g;cout=w.shape[0]//g;out_shape=p['output'][0];spatial=tuple(out_shape[2:]);outputs=[]
            for group in range(g):
                data=x[:,group*cin:(group+1)*cin];pads=tuple(v for z in reversed(pad) for v in (z,z))
                if any(pads):data=F.pad(data,pads)
                for i,(k,t) in enumerate(zip(effective,stride)):data=data.unfold(2+i,k,t)
                data=data[(slice(None),slice(None))+tuple(slice(0,o) for o in spatial)+tuple(slice(None,None,d) for d in dilation)]
                order=(0,)+tuple(range(2,nd+2))+(1,)+tuple(range(nd+2,2*nd+2))
                patches=data.permute(order).contiguous().reshape(-1,cin*prod(kernel))
                weights=w[group*cout:(group+1)*cout].reshape(cout,-1)
                bias_part=None if bias is None else bias[group*cout:(group+1)*cout]
                value=mm(patches,weights.T,bias_part).reshape(x.shape[0],*spatial,cout)
                outputs.append(value.permute(0,nd+1,*range(1,nd+1)))
            return torch.cat(outputs,dim=1).contiguous()
        raise UnsupportedCase(op)
    def cleanup():
        for child in children.values():child.cleanup()
    original=SOURCE/'math/reduce_mean_with_count';sources=[q for part in ('op_host','op_kernel','op_graph') for q in (original/part).rglob('*') if q.is_file() and q.suffix in ('.cpp','.h','.py')]
    return Candidate(Implementation(run),'ops-math::ReduceMeanWithCount(x=A,count=B,count_sum=1)',
      'Pack dot-product rows with exact FP32 widening and constant unit divisors; optional bias is one additional constant-one coordinate. Original weighted-reduction DAG and schedule are unchanged. Original grouping, transpose and convolution windows are preserved; packing and bounded stream drains are timed.',
      source_hashes(Path(__file__),Path(__file__).with_name('graph.py'),*sources)|deployment_hashes(),timing_mode='host_roundtrip',cleanup=cleanup)
