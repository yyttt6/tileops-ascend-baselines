"""Centered normalization composed from unchanged FLA sum/L2/causal APIs."""
from math import prod, sqrt
from pathlib import Path


def centered(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    from .providers import _triton_module
    scan,scan_source=_triton_module('cumsum')
    norm,norm_source=_triton_module('l2norm')
    conv,conv_source=_triton_module('causal_conv1d')
    name=case._call.signature.name
    p=case.params
    eps=p.get("eps")
    eps=torch.finfo(torch.float32).eps if eps is None else eps
    instance=name=='InstanceNormFwdOp'
    group=name=='GroupNormFwdOp'
    fused=name=='FusedAddLayerNormFwdOp'
    adaptive=name in ('AdaLayerNormFwdOp','AdaLayerNormZeroFwdOp')
    if instance and (not p.get('use_input_stats',True) or case.inputs[1] is not None):
        from .batchnorm import batchnorm
        return batchnorm(case)

    def affine(x,weight=None,bias=None):
        shape=x.shape;n=x.numel();extent=n+-n%256
        data=F.pad(x.contiguous().reshape(-1),(0,extent-n)).reshape(1,1,extent)
        weight=torch.ones((),device=x.device,dtype=x.dtype) if weight is None else weight
        w=F.pad(weight.expand(shape).contiguous().reshape(-1),(0,extent-n)).reshape(1,extent)
        b=None if bias is None else F.pad(bias.expand(shape).contiguous().reshape(-1),(0,extent-n))
        out,_=conv.causal_conv1d_fwd_impl_old(data,w,bias=b, residual=None)
        return out.reshape(-1)[:n].reshape(shape)

    def run(x,*args):
        if fused:
            residual,*args=args
            x=affine(x,bias=residual)
        if instance:
            _,_,weight,bias=args
        else:
            weight,bias,*extra=args
        if group or instance:
            groups=x.shape[1] if instance else p['num_groups']
            width=x.numel()//(x.shape[0]*groups)
        else:
            width=prod(p.get('normalized_shape',(x.shape[-1],)))
        data=x.float().contiguous().reshape(-1,width)
        sums=data;parts=width
        while parts>16384:
            chunks=(parts+4095)//4096
            padded=F.pad(sums,(0,chunks*4096-parts)).reshape(-1,4096,1)
            sums=scan.chunk_local_cumsum(padded,chunk_size=4096,output_dtype=torch.float32)[:,-1,0].reshape(data.shape[0],chunks)
            parts=chunks
        extent=1<<(parts-1).bit_length()
        padded=F.pad(sums,(0,extent-parts)).reshape(-1,extent,1)
        mean=scan.chunk_local_cumsum(padded,chunk_size=extent,scale=1.0/width,output_dtype=torch.float32)[:,parts-1,0,None]
        minus=torch.full((),-1.0,device=x.device,dtype=torch.float32)
        centered=affine(mean.expand_as(data),minus,data)
        if width<=16384:
            unit,_=norm.l2norm_fwd(centered,eps=width*eps,output_dtype=torch.float32)
        else:
            lengths=[]
            for start in range(0,width,8192):
                _,inverse=norm.l2norm_fwd(centered[:,start:start+8192].contiguous(),eps=0.0,output_dtype=torch.float32)
                _,length=norm.l2norm_fwd(inverse.reshape(-1,1),eps=0.0,output_dtype=torch.float32)
                lengths.append(length)
            _,inverse=norm.l2norm_fwd(torch.stack(lengths,dim=1),eps=width*eps,output_dtype=torch.float32)
            unit=affine(centered,inverse[:,None])
        out=affine(unit,torch.full((),sqrt(width),device=x.device,dtype=torch.float32)).reshape(x.shape)
        if group or instance:
            affine_shape=(1,x.shape[1],*(1 for _ in x.shape[2:]))
            weight=None if weight is None else weight.reshape(affine_shape)
            bias=None if bias is None else bias.reshape(affine_shape)
        out=affine(out,None if weight is None else weight.float(),None if bias is None else bias.float())
        if adaptive and extra:out=affine(out,extra[0].float())
        out=out.to(x.dtype)
        return (out,x) if fused else out

    return Candidate(Implementation(run),'FLA chunk_local_cumsum + l2norm_fwd + causal_conv1d_fwd_impl_old',
        'Native mean prefix, native mean subtraction, L2 of centered data with N*eps and native sqrt(N)/affine scaling. Wide reductions compose bounded original scans and L2 reciprocal lengths, retaining FP32 intermediate values. Original residual rounding for fused add. Group/instance axes are reshape-only. All packing and own-API arithmetic inside run; no altered kernel/schedule.',
        source_hashes(Path(__file__),scan_source,norm_source,norm_source.with_name('utils.py'),conv_source))


def lerp(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .providers import _triton_module
    conv,source=_triton_module('causal_conv1d')
    scalar=case.params.get('weight')
    def run(x,y,weight=None):
        weight=torch.full((),scalar,device=x.device,dtype=torch.float32) if weight is None else weight.float()
        shape=torch.broadcast_shapes(x.shape,y.shape,weight.shape)
        n=prod(shape);extent=n+-n%256
        def packed(value):return F.pad(value.float().expand(shape).contiguous().reshape(-1),(0,extent-n)).reshape(1,1,extent)
        xp,yp=packed(x),packed(y)
        negative=torch.full((1,extent),-1.0,device=x.device,dtype=torch.float32)
        delta,_=conv.causal_conv1d_fwd_impl_old(xp,negative,residual=yp, bias=None)
        out,_=conv.causal_conv1d_fwd_impl_old(delta,packed(weight).reshape(1,extent),residual=xp, bias=None)
        return out.reshape(-1)[:n].reshape(shape).to(x.dtype)
    return Candidate(Implementation(run),'FLA causal_conv1d_fwd_impl_old(tap/residual interpolation)',
        'Lossless FP32 broadcast/packing; original negative single-tap residual computes end-start, then original weighted tap/residual computes start+weight*delta. One final output cast.',source_hashes(Path(__file__),source))
