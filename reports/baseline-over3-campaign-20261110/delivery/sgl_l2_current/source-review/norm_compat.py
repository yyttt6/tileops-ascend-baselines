"""Alternative original API selection and complete InstanceNorm statistics."""
from pathlib import Path


def fused_add_rms(case):
    import torch
    from .norm import fused_add_rms as direct
    x,residual,weight=case.inputs
    if x.dtype!=torch.bfloat16 or x.shape[-1]!=16384:
        return direct(case)
    from dataclasses import replace
    from types import SimpleNamespace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    from benchmarks.ascend_add_cases import sgl_add_cases
    from benchmarks.ascend_sgl_large_rms import sgl_large_rms_cases
    add=sgl_add_cases(SimpleNamespace(inputs=(x,residual),params={'alpha':1}))
    rms=sgl_large_rms_cases(SimpleNamespace(inputs=(x,weight),params={'normalized_shape':[x.shape[-1]],'eps':case.params['eps']}))
    def run(x,residual,weight):
        summed=add.implementation.run(x,residual)
        return rms.implementation.run(summed,weight),summed
    return replace(rms,implementation=Implementation(run),api=add.api+' + '+rms.api,
                   adaptation='Select existing complete Add then RMSNorm APIs for the native fused kernel UB-overflow shape; retain storage-rounded residual and both outputs, time all original calls.',
                   source_sha256=add.source_sha256|rms.source_sha256|source_hashes(Path(__file__)))


def instancenorm(case):
    from .norm import instancenorm as direct
    original=direct(case)
    x,rm,rv,weight,bias=case.inputs
    if rm is None:
        return original
    import torch
    import math
    from dataclasses import replace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes,local_function,LIBRARIES
    from .additional import mul
    multiply=mul(case)
    host='python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py'
    affine=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.moe.mul_add','mul_add',host)
    nhost='python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    norm=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu_simd',nhost)
    n,c,*spatial=x.shape
    width=math.prod(spatial)
    def mean_rows(value):
        return norm(value,torch.ones(value.shape[1],device=value.device,dtype=value.dtype),None,1e-5)[1].reshape(-1)
    def run(x,rm,rv,weight,bias):
        y=original.implementation.run(x,rm,rv,weight,bias)
        packed=x.float().reshape(n*c,width)
        means=mean_rows(packed)
        centered=affine(means[:,None].expand(n*c,width).contiguous(),packed,-1.0)
        variances=mean_rows(multiply.implementation.run(centered,centered))
        mean=mean_rows(means.reshape(n,c).t().contiguous())
        var=mean_rows(variances.reshape(n,c).t().contiguous())
        momentum=case.params['momentum']
        oldm=affine(rm.reshape(c,1),torch.zeros_like(rm).reshape(c,1),1-momentum)
        newm=affine(mean.reshape(c,1),oldm,momentum).reshape_as(rm)
        oldv=affine(rv.reshape(c,1),torch.zeros_like(rv).reshape(c,1),1-momentum)
        newv=affine(var.reshape(c,1),oldv,momentum*width/(width-1)).reshape_as(rv)
        return y,newm,newv
    return replace(original,implementation=Implementation(run),
                   api=original.api+' + native mean/square/affine running statistics',
                   adaptation=original.adaptation+' Return both declared running statistics using original SGL complete spatial mean/variance and batch mean APIs; preserve all input buffers.',
                   source_sha256=original.source_sha256|multiply.source_sha256|source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))
