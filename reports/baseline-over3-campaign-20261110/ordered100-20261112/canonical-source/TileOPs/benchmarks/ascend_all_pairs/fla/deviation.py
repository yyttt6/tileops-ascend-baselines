"""Standard deviation through original FLA scan, centering and L2 APIs."""
from math import prod, sqrt
from pathlib import Path


def deviation(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .providers import _triton_module
    scan,scan_source=_triton_module('cumsum')
    norm,norm_source=_triton_module('l2norm')
    conv,conv_source=_triton_module('causal_conv1d')
    p=case.params

    def affine(data,weight,bias=None):
        count=data.numel();extent=count+-count%256
        packed=F.pad(data.reshape(-1),(0,extent-count)).reshape(1,1,extent)
        w=torch.full((1,extent),weight,device=data.device,dtype=torch.float32)
        bp=None if bias is None else F.pad(bias.expand_as(data).contiguous().reshape(-1),(0,extent-count))
        out,_=conv.causal_conv1d_fwd_impl_old(packed,w,bias=bp,residual=None)
        return out.reshape(-1)[:count].reshape(data.shape)

    def length(data):
        if data.shape[1]>16384:
            data=torch.stack([length(data[:,i:i+8192].contiguous()) for i in range(0,data.shape[1],8192)],dim=1)
        _,inverse=norm.l2norm_fwd(data,eps=0.0,output_dtype=torch.float32)
        return norm.l2norm_fwd(inverse.reshape(-1,1),eps=0.0,output_dtype=torch.float32)[1]

    def run(x):
        dims=p.get('dim')
        dims=tuple(range(x.ndim)) if dims is None else ((dims,) if isinstance(dims,int) else tuple(dims))
        dims=tuple(d%x.ndim for d in dims)
        left=tuple(d for d in range(x.ndim) if d not in dims)
        width=prod(x.shape[d] for d in dims)
        data=x.permute(left+dims).contiguous().reshape(-1,width).float()
        sums=data;parts=width
        while parts>16384:
            chunks=(parts+4095)//4096
            packed=F.pad(sums,(0,chunks*4096-parts)).reshape(-1,4096,1)
            sums=scan.chunk_local_cumsum(packed,chunk_size=4096,output_dtype=torch.float32)[:,-1,0].reshape(data.shape[0],chunks)
            parts=chunks
        extent=1<<(parts-1).bit_length()
        packed=F.pad(sums,(0,extent-parts)).reshape(-1,extent,1)
        negative_mean=scan.chunk_local_cumsum(packed,chunk_size=extent,scale=-1.0/width,output_dtype=torch.float32)[:,parts-1,0]
        centered=affine(data,1.0,negative_mean[:,None])
        value=affine(length(centered),1.0/sqrt(width-p.get('correction',1)))
        shape=tuple(1 if d in dims else x.shape[d] for d in range(x.ndim)) if p.get('keepdim',False) else tuple(x.shape[d] for d in left)
        return value.to(x.dtype).reshape(shape)

    return Candidate(Implementation(run),'FLA chunk_local_cumsum + causal_conv1d_fwd_impl_old + l2norm_fwd(rstd)',
        'Original scaled scan computes the negative mean; native bias centers each row; native L2 reciprocal outputs recover its length, then native constant tap applies 1/sqrt(N-correction). Long rows compose bounded original L2 calls. No torch arithmetic on input values or kernel/schedule modifications.',
        source_hashes(Path(__file__),scan_source,norm_source,conv_source,norm_source.with_name('utils.py')))
