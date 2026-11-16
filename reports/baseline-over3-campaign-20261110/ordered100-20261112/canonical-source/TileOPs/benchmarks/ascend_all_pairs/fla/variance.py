"""Variance via unchanged original FLA scan and causal-product APIs."""
from math import prod
from pathlib import Path


def variance(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .providers import _triton_module
    scan,scan_source=_triton_module('cumsum')
    conv,conv_source=_triton_module('causal_conv1d')
    p=case.params
    both=case._call.signature.name=='VarMeanFwdOp'
    def run(x):
        dims=p.get('dim')
        dims=tuple(range(x.ndim)) if dims is None else ((dims,) if isinstance(dims,int) else tuple(dims))
        dims=tuple(d%x.ndim for d in dims)
        left=tuple(d for d in range(x.ndim) if d not in dims)
        shape=tuple(x.shape[d] for d in left)
        width=prod(x.shape[d] for d in dims)
        data=x.permute(left+dims).contiguous().reshape(-1,width).float()
        def reduce(values,scale):
            parts=width
            while parts>16384:
                chunks=(parts+4095)//4096
                packed=F.pad(values,(0,chunks*4096-parts)).reshape(-1,4096,1)
                values=scan.chunk_local_cumsum(packed,chunk_size=4096,output_dtype=torch.float32)[:,-1,0].reshape(values.shape[0],chunks)
                parts=chunks
            extent=1<<(parts-1).bit_length()
            packed=F.pad(values,(0,extent-parts)).reshape(-1,extent,1)
            return scan.chunk_local_cumsum(packed,chunk_size=extent,scale=scale,output_dtype=torch.float32)[:,parts-1,0]
        mean=reduce(data,1.0/width)
        n=data.numel();channels=n+-n%256
        a=F.pad(mean[:,None].expand_as(data).contiguous().reshape(-1),(0,channels-n)).reshape(1,1,channels)
        b=F.pad(data.reshape(-1),(0,channels-n)).reshape(1,1,channels)
        negative=torch.full((1,channels),-1.0,device=x.device,dtype=torch.float32)
        centered,_=conv.causal_conv1d_fwd_impl_old(a,negative,residual=b, bias=None)
        squared,_=conv.causal_conv1d_fwd_impl_old(centered,centered.reshape(1,channels), bias=None, residual=None)
        variance=reduce(squared.reshape(-1)[:n].reshape(data.shape),1.0/(width-p.get('correction',1)))
        if p.get('keepdim',False):shape=tuple(1 if d in dims else x.shape[d] for d in range(x.ndim))
        result=variance.to(x.dtype).reshape(shape)
        return (result,mean.to(x.dtype).reshape(shape)) if both else result
    return Candidate(Implementation(run),'FLA chunk_local_cumsum + causal_conv1d_fwd_impl_old(centered square)',
        'Permute/reshape reduction axes; original scaled scan computes mean, original residual/tap calls center and square FP32 data, original scaled scan applies manifest correction. All transforms inside run; no torch arithmetic on values.',source_hashes(Path(__file__),scan_source,conv_source))
