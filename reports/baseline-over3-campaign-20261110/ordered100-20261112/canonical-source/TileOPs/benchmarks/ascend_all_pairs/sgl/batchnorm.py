"""Original SGL batch/spatial normalization and full running-stat outputs."""
from pathlib import Path
import math
from .norm import _candidate


def batchnorm(case):
    import torch
    from benchmarks.ascend_baselines import local_function,UnsupportedCase
    if not case.params['training']:
        raise UnsupportedCase('Original grouped LayerNorm computes input statistics; running-statistics inference is a different public contract.')
    x,rm,rv,weight,bias=case.inputs
    n,c,*spatial=x.shape
    width=n*math.prod(spatial)
    if width*x.element_size()>65536:
        raise UnsupportedCase('Original unchanged layer_norm_fwd_npu_simd public host rejects complete feature rows larger than64KiB; partial statistics would require a new reduction algorithm.')
    host='python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu_simd',host)
    order=(1,0,*range(2,x.ndim))
    from .additional import mul
    from types import SimpleNamespace
    from benchmarks.ascend_baselines import source_hashes,LIBRARIES
    from dataclasses import replace
    multiply=mul(case)
    addhost='python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py'
    affine_fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.moe.mul_add','mul_add',addhost)
    def run(x,rm,rv,weight,bias):
        packed=x.permute(order).contiguous().reshape(1,c*width)
        affine=torch.ones(c*width,device=x.device,dtype=x.dtype) if weight is None else weight[:,None].expand(c,width).contiguous().reshape(-1)
        shift=None if bias is None else bias[:,None].expand(c,width).contiguous().reshape(-1)
        y,mean,*_=fn(packed,affine,shift,case.params['eps'],group_size=width)
        running_mean=running_var=None
        if rm is not None:
            momentum=case.params['momentum']
            # Original reference declares updated statistics as outputs, even
            # though only output is named in the manifest tensor table.
            oldm=affine_fn(rm.reshape(-1,1),torch.zeros_like(rm).reshape(-1,1),1-momentum)
            running_mean=affine_fn(mean.reshape(-1,1),oldm,momentum).reshape_as(rm)
            centered=affine_fn(mean[:,None].expand(c,width).contiguous(),packed.float().reshape(c,width),-1.0)
            squared=multiply.implementation.run(centered,centered)
            var=fn(squared,torch.ones(width,device=x.device,dtype=torch.float32),None,1e-5)[1]
            oldv=affine_fn(rv.reshape(-1,1),torch.zeros_like(rv).reshape(-1,1),1-momentum)
            running_var=affine_fn(var.reshape(-1,1),oldv,momentum*width/(width-1)).reshape_as(rv)
        return y.reshape(c,n,*spatial).permute(order).contiguous(),running_mean,running_var
    native=_candidate(run,host,'sgl_kernel_npu.fla.layernorm_gated.layer_norm_fwd_npu_simd(group_size=N*spatial)',
        'Transpose channels first; each native group covers the full batch/spatial population for one channel. Repeat original per-channel affine and restore NCHW. Original SGL mean, centering, square, mean and affine calls also return both updated running-stat buffers; all inputs remain protected.')
    return replace(native,source_sha256=native.source_sha256|multiply.source_sha256|source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/addhost))
