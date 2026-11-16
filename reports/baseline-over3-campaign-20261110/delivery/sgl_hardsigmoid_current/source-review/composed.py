"""Complete arithmetic compositions of unchanged original SGL APIs."""
from pathlib import Path


def activation(case):
    import torch
    from dataclasses import replace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import local_function,source_hashes,LIBRARIES
    from .additional import maximum,minimum,mul
    high,low,times=maximum(case),minimum(case),mul(case)
    host='python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py'
    affine=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.moe.mul_add','mul_add',host)
    name=case._call.signature.name
    def apply_affine(x,y,factor):
        # Independent elements may share a row; use the already verified
        # 4096-column layout instead of one native serial row per element.
        import torch.nn.functional as F
        size=x.numel();tail=-size%4096
        xx=F.pad(x.contiguous().reshape(-1),(0,tail)).reshape(-1,4096)
        yy=F.pad(y.expand_as(x).contiguous().reshape(-1),(0,tail)).reshape(-1,4096)
        return affine(xx,yy,factor).reshape(-1)[:size].reshape_as(x)
    def run(x,weight=None):
        xf=x.float()
        zero=torch.zeros((),device=x.device,dtype=torch.float32)
        if name in ('HardsigmoidFwdOp','HardswishFwdOp'):
            shifted=apply_affine(xf,torch.full_like(xf,3),1.0)
            clipped=low.implementation.run(high.implementation.run(shifted,zero),torch.full((),6,device=x.device,dtype=torch.float32))
            if name=='HardswishFwdOp':clipped=times.implementation.run(xf,clipped)
            out=apply_affine(clipped,torch.zeros_like(clipped),1.0/6)
        else:
            positive=high.implementation.run(xf,zero)
            negative=low.implementation.run(xf,zero)
            if name=='LeakyReluFwdOp':
                out=apply_affine(negative,positive,case.params['negative_slope'])
            else:
                shape=(1,-1,*([1]*(x.ndim-2))) if weight.numel()!=1 and x.ndim>1 else ()
                scale=weight.float().reshape(shape)
                product=times.implementation.run(negative,scale)
                out=apply_affine(product,positive,1.0)
        return out.to(x.dtype)
    return replace(high,implementation=Implementation(run),api='SGL original argmax selector + scale_shift + mul_add',
                   adaptation='Lossless FP32 input/weight widening, native clamp or positive/negative selectors and original affine/product APIs; one final storage cast. All per-call conversions and launches are timed.',
                   source_sha256=high.source_sha256|low.source_sha256|times.source_sha256|source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))


def variance(case):
    import torch
    from dataclasses import replace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import local_function,source_hashes,LIBRARIES,UnsupportedCase
    from .reduction import row_layout
    from .additional import mul
    pack,width,shape=row_layout(case)
    correction=case.params.get('correction')
    correction=1 if correction is None else correction
    if width<=correction:raise UnsupportedCase('Nonpositive variance degrees of freedom have no native finite affine form.')
    nhost='python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    ahost='python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py'
    norm=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu_simd',nhost)
    affine=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.moe.mul_add','mul_add',ahost)
    times=mul(case)
    def run(x):
        packed=pack(x).float()
        unit=torch.ones(width,device=x.device,dtype=torch.float32)
        means=norm(packed,unit,None,1e-5)[1].reshape(-1,1)
        centered=affine(means.expand_as(packed).contiguous(),packed,-1.0)
        square=times.implementation.run(centered,centered)
        var=norm(square,unit,None,1e-5)[1].reshape(-1,1)
        out=affine(var,torch.zeros_like(var),width/(width-correction)).to(x.dtype).reshape(shape)
        return (out,means.to(x.dtype).reshape(shape)) if case._call.signature.name=='VarMeanFwdOp' else out
    return replace(times,implementation=Implementation(run),api='SGL original mean + mul_add centering + scale_shift square + mean + correction',
                   adaptation='Complete reduction row and original FP32 mean, center, square and mean APIs; native scalar affine degrees-of-freedom correction, all declared outputs returned. No partial reductions or schedule changes.',
                   source_sha256=times.source_sha256|source_hashes(Path(__file__),Path(__file__).with_name('reduction.py'),LIBRARIES/'sgl-kernel-npu'/nhost,LIBRARIES/'sgl-kernel-npu'/ahost))
