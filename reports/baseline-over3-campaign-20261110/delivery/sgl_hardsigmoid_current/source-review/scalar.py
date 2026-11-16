"""Constant/selector compositions entirely in original SGL arithmetic APIs."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace


def selectors(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    from .additional import maximum,minimum
    from .elementwise import neg
    x=case.inputs[0]
    native_max=maximum(case)
    native_min=minimum(case)
    native_neg=neg(SimpleNamespace(inputs=(x,),params={}))
    name=case._call.signature.name
    def run(x,lo=None,hi=None):
        if name=='AbsFwdOp':return native_max.implementation.run(native_neg.implementation.run(x),x)
        if name=='ReluFwdOp':return native_max.implementation.run(x,torch.zeros((),device=x.device,dtype=x.dtype))
        if name in ('ClampScalarFwdOp','HardtanhFwdOp'):
            a=case.params.get('min',case.params.get('min_val'))
            b=case.params.get('max',case.params.get('max_val'))
            lo=None if a is None else torch.tensor(a,device=x.device,dtype=x.dtype)
            hi=None if b is None else torch.tensor(b,device=x.device,dtype=x.dtype)
        out=x
        if lo is not None:out=native_max.implementation.run(out,lo)
        if hi is not None:out=native_min.implementation.run(out,hi)
        return out
    return replace(native_max,implementation=Implementation(run),api=native_max.api+' + original mul_add sign inversion',
        adaptation='Constant bound tensors and broadcast/stack rows; original SGL max/min/sign APIs perform every selector computation; no external comparisons or clamp.',
        source_sha256=native_max.source_sha256|native_min.source_sha256|native_neg.source_sha256|source_hashes(Path(__file__)))


def lerp_scalar(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes,UnsupportedCase
    weight=case.params['weight']
    if weight!=0.5:raise UnsupportedCase('Only exact midpoint uses complete original native mean without additional rounded arithmetic')
    host='python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu_simd',host)
    def run(x,y):
        shape=torch.broadcast_shapes(x.shape,y.shape)
        pairs=torch.stack((x.expand(shape),y.expand(shape)),dim=-1).reshape(-1,2)
        return fn(pairs,torch.ones(2,device=x.device,dtype=x.dtype),None,1e-5)[1].to(x.dtype).reshape(shape)
    return Candidate(Implementation(run),'sgl_kernel_npu.fla.layernorm_gated.layer_norm_fwd_npu_simd()[1]',
        'Exact midpoint specialization: broadcast each pair into a complete two-element native mean; original mean remains FP32 until final output cast.',
        source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))


def sum_reduce(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes
    from .reduction import row_layout
    pack,width,shape=row_layout(case)
    normhost='python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    mulhost='python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py'
    norm=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu_simd',normhost)
    mul=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.moe.mul_add','mul_add',mulhost)
    outdtype=getattr(torch,case.params.get('dtype') or str(case.inputs[0].dtype).split('.')[-1])
    def run(x):
        value=norm(pack(x),torch.ones(width,device=x.device,dtype=x.dtype),None,1e-5)[1].reshape(-1,1)
        return mul(value,torch.zeros_like(value),width).to(outdtype).reshape(shape)
    return Candidate(Implementation(run),'sgl_kernel_npu::layer_norm_fwd_npu_simd()[1] + mul_add',
        'Complete reduction row in original mean API; preserve FP32 mean, original native constant width multiply, final declared output cast. No external sum arithmetic.',
        source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/normhost,LIBRARIES/'sgl-kernel-npu'/mulhost),
        limitations=('Reduction rounding differs from standalone sum, so every case must pass the unchanged original tolerance.',))
