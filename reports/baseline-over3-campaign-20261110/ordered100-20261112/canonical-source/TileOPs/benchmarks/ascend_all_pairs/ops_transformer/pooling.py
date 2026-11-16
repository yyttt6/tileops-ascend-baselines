"""Pool windows are packed; original FP32 GroupedMatmul performs every mean."""
from pathlib import Path


def pooling(case):
    import itertools,math,torch
    from collections import defaultdict
    from dataclasses import replace
    from .adapters import _candidate,_float_product
    from .native import GraphCall
    from benchmarks.ascend_baselines import source_hashes
    from .hashing import source_hashes
    op=case._call.signature.name;p=case.params;binding=GraphCall('ops-transformer')
    def product(rows,divisor):
        coeff=torch.full((rows.shape[-1],1),1/divisor,dtype=torch.float32,device=rows.device)
        return _float_product(binding,rows.float(),coeff)[:,0]
    def run(x,*metadata):
        if op=='MeanPoolingFwdOp':
            # Shared helper only resolves integer chunk boundaries; no oracle math.
            slices=case.workload.chunk_slices(x,*metadata)
            if not slices:return x.new_empty((x.shape[0],0,*x.shape[2:]))
            groups=defaultdict(list)
            for i,(start,end) in enumerate(slices):groups[end-start].append((i,start,end))
            out=torch.empty((x.shape[0],len(slices),*x.shape[2:]),dtype=x.dtype,device=x.device)
            for length,items in groups.items():
                rows=torch.stack([x[:,start:end].movedim(1,-1) for _,start,end in items],1)
                reduced=product(rows.contiguous().reshape(-1,length),length).reshape(x.shape[0],len(items),*x.shape[2:]).to(x.dtype)
                ids=torch.tensor([item[0] for item in items],dtype=torch.int64,device=x.device)
                out.index_copy_(1,ids,reduced)
            return out
        if op=='AdaptiveAvgPool2dFwdOp':
            spatial=p['output_size'];h,w=x.shape[-2:];groups=defaultdict(list)
            for i,j in itertools.product(range(spatial[0]),range(spatial[1])):
                bounds=(i*h//spatial[0],-(-((i+1)*h)//spatial[0]),j*w//spatial[1],-(-((j+1)*w)//spatial[1]))
                groups[(bounds[1]-bounds[0])*(bounds[3]-bounds[2])].append((i*spatial[1]+j,bounds))
            leading=x.shape[:-2];out=torch.empty((*leading,math.prod(spatial)),dtype=x.dtype,device=x.device)
            for width,items in groups.items():
                rows=torch.stack([x[...,a:b,c:d].contiguous().flatten(-2) for _,(a,b,c,d) in items],-2)
                reduced=product(rows.reshape(-1,width),width).reshape(*leading,len(items)).to(x.dtype)
                ids=torch.tensor([i for i,_ in items],dtype=torch.int64,device=x.device)
                out.index_copy_(-1,ids,reduced)
            return out.reshape(*leading,*spatial)
        rank=x.ndim-2
        tup=lambda v: (v,)*rank if isinstance(v,int) else tuple(v)
        kernel=tup(p['kernel_size']);stride=tup(p['stride'] if p['stride'] is not None else p['kernel_size']);padding=tup(p['padding'])
        output=[]
        for size,k,s,pad in zip(x.shape[2:],kernel,stride,padding,strict=True):
            n=(size+2*pad-k+(s-1 if p['ceil_mode'] else 0))//s+1
            if p['ceil_mode'] and (n-1)*s>=size+pad:n-=1
            output.append(n)
        right=[max(pad,(n-1)*s+k-size-pad) for n,s,k,size,pad in zip(output,stride,kernel,x.shape[2:],padding,strict=True)]
        padtuple=tuple(v for left,end in reversed(list(zip(padding,right))) for v in (left,end))
        padded=torch.nn.functional.pad(x,padtuple)
        sizes=(*x.shape[:2],*output,*kernel)
        strides=(*padded.stride()[:2],*[v*s for v,s in zip(padded.stride()[2:],stride)],*padded.stride()[2:])
        windows=padded.as_strided(sizes,strides).contiguous().reshape(x.shape[0],x.shape[1],math.prod(output),math.prod(kernel))
        groups=defaultdict(list)
        for flat,coord in enumerate(itertools.product(*[range(n) for n in output])):
            count=1
            for i,s,pad,k,size in zip(coord,stride,padding,kernel,x.shape[2:],strict=True):
                start=i*s-pad;lo=-pad if p['count_include_pad'] else 0;hi=size+pad if p['count_include_pad'] else size
                count*=max(0,min(start+k,hi)-max(start,lo))
            divisor=p.get('divisor_override') or count
            groups[divisor].append(flat)
        out=torch.empty((*x.shape[:2],math.prod(output)),dtype=x.dtype,device=x.device)
        for divisor,positions in groups.items():
            ids=torch.tensor(positions,dtype=torch.int64,device=x.device)
            rows=windows.index_select(2,ids).reshape(-1,math.prod(kernel))
            values=product(rows,divisor).reshape(*x.shape[:2],len(positions)).to(x.dtype)
            out.index_copy_(2,ids,values)
        return out.reshape(*x.shape[:2],*output)
    result=_candidate(binding,run,'GroupedMatmul','gmm/grouped_matmul','Pool windows are packed with exact padding, ceil boundaries, ragged/adaptive bounds and per-window divisors. Original native FP32 GEMM with constant reciprocal coefficients performs complete mean; exact input widening, copies, index metadata and output casts all occur in run. No Torch reduction or pooling calculation.',host=op=='MeanPoolingFwdOp')
    return replace(result,source_sha256=result.source_sha256|source_hashes(Path(__file__)))
