"""Complete training BatchNorm and backward through original FLA primitives."""
from math import sqrt
from pathlib import Path


def batchnorm(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    from .providers import _triton_module
    scan,scan_source=_triton_module('cumsum')
    conv,conv_source=_triton_module('causal_conv1d')
    norm,norm_source=_triton_module('l2norm')
    p=case.params
    backward=case._call.signature.name=='BatchNormBwdOp'
    training=p.get('training',p.get('use_input_stats',False))
    instance_training=case._call.signature.name=='InstanceNormFwdOp' and training
    eps=p.get('eps')
    eps=torch.finfo(torch.float32).eps if eps is None else eps

    def affine(x,weight=1.0,bias=None):
        shape=x.shape;count=x.numel();extent=count+-count%256
        data=F.pad(x.contiguous().reshape(-1),(0,extent-count)).reshape(1,1,extent)
        if not isinstance(weight,torch.Tensor):
            weight=torch.full((),weight,device=x.device,dtype=torch.float32)
        weights=F.pad(weight.expand(shape).contiguous().reshape(-1),(0,extent-count)).reshape(1,extent)
        bp=None if bias is None else F.pad(bias.expand(shape).contiguous().reshape(-1),(0,extent-count))
        out,_=conv.causal_conv1d_fwd_impl_old(data,weights,bias=bp,residual=None)
        return out.reshape(-1)[:count].reshape(shape)

    def reduce(data,scale=1.0):
        rows,width=data.shape
        # Original API composition limits the size of a single prefix block.
        # Intermediate sums remain FP32; the requested scale is applied once.
        while width>16384:
            chunk=4096;parts=(width+chunk-1)//chunk
            packed=F.pad(data,(0,parts*chunk-width)).reshape(-1,chunk,1)
            data=scan.chunk_local_cumsum(packed,chunk_size=chunk,output_dtype=torch.float32)[:,-1,0].reshape(rows,parts)
            width=parts
        extent=1<<(width-1).bit_length()
        packed=F.pad(data,(0,extent-width)).reshape(-1,extent,1)
        return scan.chunk_local_cumsum(packed,chunk_size=extent,scale=scale,output_dtype=torch.float32)[:,width-1,0]

    def normalize(centered,width):
        if width<=16384:
            unit,_=norm.l2norm_fwd(centered,eps=width*eps,output_dtype=torch.float32)
            return affine(unit,sqrt(width))
        # L2 returns rstd. L2 of a positive scalar returns its reciprocal,
        # hence rstd(rstd(chunk)) is the chunk's L2 norm, including zero chunks.
        lengths=[]
        for start in range(0,width,8192):
            _,inverse=norm.l2norm_fwd(centered[:,start:start+8192].contiguous(),eps=0.0,output_dtype=torch.float32)
            _,length=norm.l2norm_fwd(inverse.reshape(-1,1),eps=0.0,output_dtype=torch.float32)
            lengths.append(length)
        _,inverse=norm.l2norm_fwd(torch.stack(lengths,dim=1),eps=width*eps,output_dtype=torch.float32)
        return affine(affine(centered,inverse[:,None]),sqrt(width))

    def packed(x):
        if instance_training:return x.float().contiguous().reshape(x.shape[0]*x.shape[1],-1)
        return x.float().movedim(1,0).contiguous().reshape(x.shape[1],-1)

    def unpacked(value,x):
        if instance_training:return value.reshape(x.shape).to(x.dtype)
        return value.reshape(x.shape[1],x.shape[0],*x.shape[2:]).movedim(0,1).contiguous().to(x.dtype)

    def run(first,*args):
        if backward:
            grad=first;x,weight,mean,rstd=args
            data=packed(x);delta=packed(grad);width=data.shape[1]
            centered=affine(data,bias=affine(mean[:,None],-1.0))
            xhat=affine(centered,rstd[:,None])
            grad_bias=reduce(delta)
            grad_weight=reduce(affine(delta,xhat))
            inner=affine(delta,float(width),affine(grad_bias[:,None],-1.0))
            inner=affine(xhat,affine(grad_weight[:,None],-1.0),inner)
            factor=affine(affine(weight,rstd),1.0/width)
            return unpacked(affine(inner,factor[:,None]),x),grad_weight,grad_bias
        x=first;running_mean,running_var,weight,bias=args
        data=packed(x);width=data.shape[1]
        if not training:
            # The original API accepts scalar epsilon. Pass each supplied
            # variance directly as that API argument, and a constant sqrt(eps)
            # as its one-element input: the native rstd is 1/sqrt(var+eps).
            # Only a copy to Python scalars occurs on input data; no host value
            # arithmetic or cached statistics. Host-roundtrip timing includes it.
            values=running_var.float().cpu().tolist()
            seed=torch.full((1,1),sqrt(eps),device=x.device,dtype=torch.float32)
            inverse=torch.cat([norm.l2norm_fwd(seed,eps=value,output_dtype=torch.float32)[1] for value in values])
            centered=affine(data,bias=affine(running_mean.float()[:,None],-1.0))
            result=affine(centered,inverse[:,None])
            result=affine(result,1.0 if weight is None else weight.float()[:,None],None if bias is None else bias.float()[:,None])
            return unpacked(result,x),running_mean.clone(),running_var.clone()
        mean=reduce(data,1.0/width)
        centered=affine(data,bias=affine(mean[:,None],-1.0))
        result=normalize(centered,width)
        aw=1.0 if weight is None else (weight.float().repeat(x.shape[0]) if instance_training else weight.float())[:,None]
        ab=None if bias is None else (bias.float().repeat(x.shape[0]) if instance_training else bias.float())[:,None]
        result=affine(result,aw,ab)
        next_mean=next_var=None
        if running_mean is not None:
            momentum=p.get('momentum',0.1)
            unbiased=reduce(affine(centered,centered),1.0/(width-1))
            if instance_training:
                mean=reduce(mean.reshape(x.shape[0],x.shape[1]).t().contiguous(),1.0/x.shape[0])
                unbiased=reduce(unbiased.reshape(x.shape[0],x.shape[1]).t().contiguous(),1.0/x.shape[0])
            next_mean=affine(mean,momentum,affine(running_mean.float(),1.0-momentum))
            next_var=affine(unbiased,momentum,affine(running_var.float(),1.0-momentum))
        return unpacked(result,x),next_mean,next_var

    return Candidate(Implementation(run),'FLA chunk_local_cumsum + l2norm_fwd + causal_conv1d_fwd_impl_old',
        'Channels become independent rows; tracked InstanceNorm uses independent batch-channel rows and native batch-averaged running statistics. All centering, normalization, gradients and running-stat updates execute through original FLA scan/L2/tap+bias APIs in FP32. Complete output tuples and private updated running statistics. Inference passes each running variance directly as native L2 epsilon with constant input sqrt(manifest epsilon); current variance is copied to scalar API arguments inside run, no host arithmetic on input values. Long reductions compose bounded original API calls; all packing and metadata inside run.',
        source_hashes(Path(__file__),scan_source,conv_source,norm_source,norm_source.with_name('utils.py')))
