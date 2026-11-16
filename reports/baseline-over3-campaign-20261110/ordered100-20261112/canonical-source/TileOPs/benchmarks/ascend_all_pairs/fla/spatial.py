"""Copy-only window packing into original FLA cumulative/causal primitives."""
from itertools import product
from math import prod
from pathlib import Path


def _tuple(value, n):
    return (value,)*n if isinstance(value,int) else tuple(value)


def windows(x,kernel,stride,padding,dilation,ceil=False):
    import torch.nn.functional as F
    n=x.ndim-2
    output=[(x.shape[2+i]+2*padding[i]-dilation[i]*(kernel[i]-1)-1+(stride[i]-1 if ceil else 0))//stride[i]+1 for i in range(n)]
    for i in range(n):
        if ceil and (output[i]-1)*stride[i]>=x.shape[i+2]+padding[i]:
            output[i]-=1
    right=[max(padding[i],(output[i]-1)*stride[i]+dilation[i]*(kernel[i]-1)+1-x.shape[i+2]-padding[i]) for i in range(n)]
    pads=tuple(j for i in reversed(range(n)) for j in (padding[i],right[i]))
    padded=F.pad(x,pads)
    shape=(*x.shape[:2],*output,*kernel)
    strides=(*padded.stride()[:2],*(padded.stride()[2+i]*stride[i] for i in range(n)),*(padded.stride()[2+i]*dilation[i] for i in range(n)))
    return padded.as_strided(shape,strides),tuple(output)


def average_pool(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .providers import _triton_module
    module,source=_triton_module("cumsum")
    n=int(case._call.signature.name[len("AvgPool")])
    unbatched=case.inputs[0].ndim==n+1
    p=case.params
    kernel=_tuple(p["kernel_size"],n)
    stride=_tuple(p.get("stride") or kernel,n)
    padding=_tuple(p.get("padding",0),n)
    width=prod(kernel)
    extent=1<<(width-1).bit_length()

    def run(x):
        if unbatched:x=x.unsqueeze(0)
        patches,outshape=windows(x,kernel,stride,padding,(1,)*n,p.get("ceil_mode",False))
        patches=patches.contiguous().reshape(x.shape[0],x.shape[1],-1,width)
        groups={}
        for position,coord in enumerate(product(*(range(s) for s in outshape))):
            if p.get("divisor_override") is not None:
                divisor=p["divisor_override"]
            else:
                include=p.get("count_include_pad",True)
                divisor=prod(max(0,min(c*st-pad+k,size+pad if include else size)-max(c*st-pad,-pad if include else 0)) for c,st,pad,k,size in zip(coord,stride,padding,kernel,x.shape[2:]))
            groups.setdefault(divisor,[]).append(position)
        result=torch.empty((*x.shape[:2],prod(outshape)),device=x.device,dtype=x.dtype)
        for divisor,indices in groups.items():
            part=patches[:,:,indices,:].contiguous()
            part=F.pad(part,(0,extent-width)).reshape(-1,extent,1)
            out=module.chunk_local_cumsum(part,chunk_size=extent,scale=1.0/divisor,output_dtype=x.dtype)[:,width-1,0]
            result[:,:,indices]=out.reshape(x.shape[0],x.shape[1],len(indices))
        result=result.reshape(*x.shape[:2],*outshape)
        return result.squeeze(0) if unbatched else result

    return Candidate(Implementation(run),"FLA chunk_local_cumsum(scale=1/divisor, final prefix)",
        "Copy-only padded strided window packing; group windows by exact manifest divisor including boundary/ceil/exclude-padding semantics; original cumsum and scale compute each mean.",source_hashes(Path(__file__),source))


def convolution(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    from .providers import _triton_module
    module,source=_triton_module("causal_conv1d")
    x,w,*_=case.inputs
    n=w.ndim-2
    unbatched=x.ndim==n+1
    p=case.params
    kernel=tuple(w.shape[2:])
    stride=_tuple(p.get("stride",1),n)
    padding=_tuple(p.get("padding",0),n)
    dilation=_tuple(p.get("dilation",1),n)
    groups=p.get("groups",1)
    width=w.shape[1]*prod(kernel)
    extent=1<<(width-1).bit_length()
    cout=w.shape[0]
    channels=cout+-cout%256
    cube=None
    hashes=source_hashes(Path(__file__),source)
    if x.dtype==torch.bfloat16:
        from .cube_matmul import make_matmul
        from .native_affine import make_affine
        cube,cube_hashes=make_matmul(torch.float32)
        affine,affine_hashes=make_affine()
        hashes.update(cube_hashes|affine_hashes)

    def run(x,w,bias=None):
        if unbatched:x=x.unsqueeze(0)
        patches,outshape=windows(x,kernel,stride,padding,dilation)
        # Copy the <=8-dimensional window view before splitting channel groups.
        # Flatten contiguous spatial/kernel dimensions to avoid a 9D copy for 3D.
        grouped=patches.contiguous().reshape(x.shape[0],groups,w.shape[1],prod(outshape),prod(kernel))
        values=grouped.permute(0,3,2,4,1).contiguous().reshape(-1,width,groups)
        if cube is not None:
            output=torch.empty((values.shape[0],cout),device=x.device,dtype=x.dtype)
            for group in range(groups):
                first=group*(cout//groups);last=(group+1)*(cout//groups)
                value=cube(values[:,:,group].contiguous(),w[first:last].reshape(last-first,width).t().contiguous())
                if bias is not None:value=affine(value,1.0,bias[first:last].float()[None,:])
                output[:,first:last]=value.to(x.dtype)
            result=output.reshape(x.shape[0],*outshape,cout).movedim(-1,1).contiguous()
            return result.squeeze(0) if unbatched else result
        # Each packed row is an independent native convolution batch. Limit
        # temporary expansion to 64 MiB per input, without changing the native
        # tile schedule or the order of tap accumulation within an output.
        tap_limit=64 if values.shape[0]==1 else 16
        accumulated=width>tap_limit or bias is not None
        item_bytes=4 if accumulated else x.element_size()
        row_count=max(1,(64*1024*1024)//(channels*min(extent,tap_limit)*item_bytes))
        pieces=[]
        if accumulated:
            for row_first in range(0,values.shape[0],row_count):
                rows=values[row_first:row_first+row_count]
                accumulator=None
                for first in range(0,width,tap_limit):
                    stop=min(width,first+tap_limit)
                    count=stop-first
                    extent_part=1<<(count-1).bit_length()
                    part=rows[:,first:stop].repeat_interleave(cout//groups,dim=-1).float()
                    part=F.pad(part,(0,channels-cout,extent_part-count,0))
                    taps=w.reshape(cout,width)[:,first:stop].t().contiguous().float()
                    taps=F.pad(taps,(0,channels-cout,extent_part-count,0))
                    residual=None
                    if accumulator is not None:
                        residual=torch.zeros_like(part)
                        residual[:,-1,:]=accumulator
                    # Independent spatial rows may also occupy independent
                    # convolution channels. Keep the native batch at one;
                    # repeat each row's identical filters rather than altering
                    # the original host's BT/BD or its kernel schedule.
                    rows_count=part.shape[0]
                    data=part.permute(1,0,2).contiguous().reshape(1,extent_part,rows_count*channels)
                    weights=taps.repeat(1,rows_count)
                    residual=None if residual is None else residual.permute(1,0,2).contiguous().reshape(data.shape)
                    out,_=module.causal_conv1d_fwd_impl_old(data,weights,bias=None,residual=residual)
                    accumulator=out[0,-1].reshape(rows_count,channels).contiguous()
                if bias is not None:
                    data=accumulator.contiguous().reshape(1,1,-1)
                    weights=torch.ones((1,data.shape[-1]),device=x.device,dtype=torch.float32)
                    bp=F.pad(bias.float(),(0,channels-cout)).repeat(rows_count)
                    out,_=module.causal_conv1d_fwd_impl_old(data,weights,bias=bp,residual=None)
                    accumulator=out.reshape(rows_count,channels)
                pieces.append(accumulator[:,:cout].to(x.dtype))
        else:
            taps=w.reshape(cout,width).t().contiguous()
            taps=F.pad(taps,(0,channels-cout,extent-width,0))
            bp=None if bias is None else F.pad(bias,(0,channels-cout))
            for row_first in range(0,values.shape[0],row_count):
                part=values[row_first:row_first+row_count].repeat_interleave(cout//groups,dim=-1)
                part=F.pad(part,(0,channels-cout,extent-width,0))
                out,_=module.causal_conv1d_fwd_impl_old(part,taps,bias=bp,residual=None)
                pieces.append(out[:,-1,:cout].contiguous())
        result=torch.cat(pieces,dim=0).reshape(x.shape[0],*outshape,cout).movedim(-1,1).contiguous()
        return result.squeeze(0) if unbatched else result

    return Candidate(Implementation(run),"FLA ChunkFwdH final-state matrix product of convolution windows" if cube is not None else "FLA causal_conv1d_fwd_impl_old(window dot product)",
        "Pack independent spatial windows by copy/stride/pad/crop, preserving groups and original tap order. BF16 uses original ChunkFwdH matrix products with FP32 final state, native FP32 bias and final output cast. Other dtypes compose original 16-tap calls (up to64 for one row), native FP32 residual accumulation and final bias. Bound duplicate packing, preserve original BT/BD and device schedule.",hashes)


def adaptive_average_pool(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .providers import _triton_module
    module,source=_triton_module("cumsum")
    target=case.params["output_size"]

    def run(x):
        outshape=tuple(s if t is None else t for s,t in zip(x.shape[-2:],target))
        result=torch.empty((*x.shape[:-2],*outshape),device=x.device,dtype=x.dtype)
        for i,j in product(range(outshape[0]),range(outshape[1])):
            a=i*x.shape[-2]//outshape[0]
            b=((i+1)*x.shape[-2]+outshape[0]-1)//outshape[0]
            c=j*x.shape[-1]//outshape[1]
            d=((j+1)*x.shape[-1]+outshape[1]-1)//outshape[1]
            width=(b-a)*(d-c)
            extent=1<<(width-1).bit_length()
            part=x[...,a:b,c:d].contiguous().reshape(-1,width)
            part=F.pad(part,(0,extent-width)).reshape(-1,extent,1)
            value=module.chunk_local_cumsum(part,chunk_size=extent,scale=1.0/width,output_dtype=x.dtype)[:,width-1,0]
            result[...,i,j]=value.reshape(x.shape[:-2])
        return result

    return Candidate(Implementation(run),"FLA chunk_local_cumsum(adaptive window mean)",
        "Exact adaptive floor/ceil window bounds; copy/reshape/pad each window into a native chunk and select its scaled final prefix.",source_hashes(Path(__file__),source))


def mean_pooling(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .providers import _triton_module
    module,source=_triton_module("cumsum")
    chunk=case.params["chunk_size"]

    def run(x,offsets=None,indices=None):
        if offsets is None:
            spans=[(a,min(x.shape[1],a+chunk)) for a in range(0,x.shape[1],chunk)]
        else:
            boundaries=offsets.cpu().tolist()
            coords=indices.cpu().tolist()
            spans=[(boundaries[seq]+local*chunk,min(boundaries[seq+1],boundaries[seq]+(local+1)*chunk)) for seq,local in coords]
        groups={}
        for index,(a,b) in enumerate(spans):
            groups.setdefault(b-a,[]).append((index,a,b))
        result=torch.empty((x.shape[0],len(spans),*x.shape[2:]),device=x.device,dtype=x.dtype)
        for width,entries in groups.items():
            extent=1<<(width-1).bit_length()
            packed=torch.stack([x[:,a:b] for _,a,b in entries],dim=1).movedim(2,-1).contiguous()
            data=F.pad(packed,(0,extent-width)).reshape(-1,extent,1)
            values=module.chunk_local_cumsum(data,chunk_size=extent,scale=1.0/width,output_dtype=x.dtype)[:,width-1,0]
            result[:,[i for i,_,_ in entries]]=values.reshape(x.shape[0],len(entries),*x.shape[2:])
        return result

    return Candidate(Implementation(run),"FLA chunk_local_cumsum(sequence chunk mean)",
        "Preserve supplied ragged offsets and chunk index order; group valid chunks by actual width; copy/pad and native scale+prefix select. Metadata refreshed inside run.",source_hashes(Path(__file__),source))


def gemm(case):
    from types import SimpleNamespace
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    batched=case._call.signature.name=="BmmFwdOp"
    p=case.params
    def matrices(a,b):
        return (a.transpose(-1,-2) if p.get('trans_a',False) else a,
                b if p.get('trans_b',False) else b.transpose(-1,-2))
    a,b=matrices(*case.inputs)
    sample_a,sample_b=(a[0],b[0]) if batched else (a,b)
    view=SimpleNamespace(inputs=(sample_a.unsqueeze(-1),sample_b.unsqueeze(-1)),
                         params=dict(stride=1,padding=0,dilation=1,groups=1))
    native=convolution(view)
    def run(a,b):
        a,weight=matrices(a,b)
        if not batched:
            return native.implementation.run(a.unsqueeze(-1),weight.unsqueeze(-1))[...,0]
        return torch.stack([native.implementation.run(aa.unsqueeze(-1),ww.unsqueeze(-1))[...,0] for aa,ww in zip(a,weight)])
    return Candidate(Implementation(run),native.api+"(matrix rows as independent 1D windows)",
        "Map A rows to independent convolution windows and B rows to original causal weights; wide K uses native FP32 tap/residual composition, with one final output cast; transpose/reshape and batch loop only outside FLA arithmetic.",
        native.source_sha256|source_hashes(Path(__file__)))
