"""Native affine/product specializations with complete declared outputs."""
from pathlib import Path


def dequant(case):
    import torch
    from dataclasses import replace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    from .additional import mul
    native=mul(case)
    name=case._call.signature.name
    output_dtype=getattr(torch,case.params['out_dtype'])
    def run(q,scale):
        if name=='INT8DequantPerChannelFwdOp':scale=scale[:,None]
        elif name=='INT8DequantPerBlockFwdOp':scale=scale.repeat_interleave(128,dim=1)[:,:q.shape[1]]
        return native.implementation.run(q.float(),scale).to(output_dtype)
    return replace(native,implementation=Implementation(run),
                   adaptation='Lossless INT8-to-FP32 widening, per-tensor/per-row/per128-element scale layout, original native scale_shift multiplication and declared output conversion. All scale expansion and conversions are per-call.',
                   source_sha256=native.source_sha256|source_hashes(Path(__file__)))


def _affine(case):
    from benchmarks.ascend_baselines import local_function,LIBRARIES,source_hashes
    host='python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.moe.mul_add','mul_add',host)
    def run(x,y,factor):
        import torch
        import torch.nn.functional as F
        shape=torch.broadcast_shapes(x.shape,y.shape)
        xx=x.expand(shape).contiguous().reshape(-1)
        yy=y.expand(shape).contiguous().reshape(-1)
        size=xx.numel();tail=-size%4096
        return fn(F.pad(xx,(0,tail)).reshape(-1,4096),F.pad(yy,(0,tail)).reshape(-1,4096),factor).reshape(-1)[:size].reshape(shape)
    return run,source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host)


def tanh(case):
    import torch
    from dataclasses import replace
    from benchmarks.api import Implementation
    from .elementwise import sigmoid
    native=sigmoid(case)
    affine,hashes=_affine(case)
    def run(x):
        xx=x.float()
        doubled=affine(xx,torch.zeros_like(xx),2.0)
        probability=native.implementation.run(doubled)
        return affine(probability,torch.full_like(probability,-1),2.0).to(x.dtype)
    return replace(native,implementation=Implementation(run),
                   api=native.api+' + original mul_add',
                   adaptation='Complete tanh identity2*sigmoid(2*x)-1 using original native affine and sigmoid APIs, lossless FP32 widening and final output cast. No kernel expression/schedule or tolerance changed.',
                   source_sha256=native.source_sha256|hashes)


def lerp(case):
    from dataclasses import replace
    from benchmarks.api import Implementation
    from .additional import mul
    native=mul(case)
    affine,hashes=_affine(case)
    def run(x,end,weight):
        xx,ee,ww=x.float(),end.float(),weight.float()
        delta=affine(xx,ee,-1.0)
        scaled=native.implementation.run(delta,ww)
        return affine(scaled,xx,1.0).to(x.dtype)
    return replace(native,implementation=Implementation(run),
                   api=native.api+' + original mul_add',
                   adaptation='Full broadcast tensor interpolation x+weight*(end-x), exclusively original native product/affine APIs in losslessly widenedFP32; final declared storage cast. Numeric equivalence must pass the unchanged original tolerance.',
                   source_sha256=native.source_sha256|hashes)


def lerp_scalar(case):
    if case.params['weight'] == 0.5:
        from .scalar import lerp_scalar as midpoint
        return midpoint(case)
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    affine,hashes=_affine(case)
    weight=case.params['weight']
    def run(x,end):
        xx,ee=x.float(),end.float()
        delta=affine(xx,ee,-1.0)
        # Match the stable endpoint form selected by scalar torch.lerp.
        output=affine(delta,xx,weight) if abs(weight)<0.5 else affine(delta,ee,-(1.0-weight))
        return output.to(x.dtype)
    return Candidate(Implementation(run),'sgl_kernel_npu.moe.mul_add.mul_add (scalar interpolation)',
        'Broadcast and losslessly widen both endpoints to FP32; original native affine computes their difference and stable endpoint interpolation, then one declared storage conversion. Scalar weight arithmetic is shape-independent host metadata; no external tensor arithmetic.',hashes)
