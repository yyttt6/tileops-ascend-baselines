"""Further adapters selecting original SGL arithmetic APIs."""
from pathlib import Path


def mul(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes
    host='python/sgl_kernel_npu/sgl_kernel_npu/norm/scale_shift.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.norm.scale_shift','fused_scale_shift',host)
    def run(x,y):
        shape=torch.broadcast_shapes(x.shape,y.shape)
        xx=x.expand(shape).contiguous().reshape(1,1,-1)
        yy=y.expand(shape).contiguous().reshape(-1)
        # Full-length shift selects the native scale_constant-capable API path.
        zeros=torch.zeros_like(xx)
        return fn(xx,yy,zeros,scale_constant=0.0).reshape(shape)
    return Candidate(Implementation(run),'sgl_kernel_npu.norm.scale_shift.fused_scale_shift(scale_constant=0)',
                     'Broadcast both operands and flatten as one native channel axis; zero full-size shift selects original x*scale API path. No external multiplication.',
                     source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))


def maximum(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes
    host='python/sgl_kernel_npu/sgl_kernel_npu/sample/argmax_softmax_prob.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.sample.argmax_softmax_prob','argmax_softmax_prob_fused',host)
    def run(x,y):
        shape=torch.broadcast_shapes(x.shape,y.shape)
        pairs=torch.stack((x.expand(shape),y.expand(shape)),dim=-1).reshape(-1,2)
        indices,_=fn(pairs)
        return pairs.gather(1,indices[:,None]).reshape(shape)
    return Candidate(Implementation(run),'sgl_kernel_npu.sample.argmax_softmax_prob.argmax_softmax_prob_fused',
                     'Broadcast and stack each independent pair as a two-element reduction row; native argmax then gather selected original values.',
                     source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host),
                     limitations=('The inspected argmax uses finite frozen inputs; nonfinite maximum semantics are not generalized.',))


def fused_add_layernorm(case):
    from dataclasses import replace
    from types import SimpleNamespace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    from benchmarks.ascend_add_cases import sgl_add_cases
    from .norm import layernorm
    x,residual,weight,bias=case.inputs
    add=sgl_add_cases(SimpleNamespace(inputs=(x,residual),params={'alpha':1}))
    norm=layernorm(SimpleNamespace(inputs=(x,weight,bias),params={'normalized_shape':[x.shape[-1]],'eps':case.params['eps']}))
    def run(x,residual,weight,bias):
        summed=add.implementation.run(x,residual)
        return norm.implementation.run(summed,weight,bias),summed
    return replace(norm,implementation=Implementation(run),api=add.api+' + '+norm.api,
                   adaptation='Original SGL residual sum rounded to storage dtype, then original SGL LayerNorm; preserve both declared outputs and time all calls.',
                   source_sha256=norm.source_sha256|add.source_sha256|source_hashes(Path(__file__)))


def ada_layernorm(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes
    host='python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu_simd',host)
    def run(x,scale,shift):
        width=x.shape[-1]
        xx=x.contiguous().reshape(-1,width)
        ss=scale.expand_as(x).contiguous().reshape(-1,width)
        bb=shift.expand_as(x).contiguous().reshape(-1,width)
        # Native group affine varies by group, so one row can contain all
        # independent normalization groups without a Python row launch loop.
        return fn(xx.reshape(1,-1),ss.reshape(-1),bb.reshape(-1),case.params['eps'],group_size=width)[0].reshape_as(x)
    return Candidate(Implementation(run),'sgl_kernel_npu.fla.layernorm_gated.layer_norm_fwd_npu_simd(group_size)',
                     'Flatten independent rows into native normalization groups, bind per-row scale/shift as per-group affine. Complete mean/variance and affine stay native.',
                     source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))


def mean_pooling(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes
    host='python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu_simd',host)
    chunk=case.params['chunk_size']
    def run(x,offsets=None,indices=None):
        if offsets is None:
            spans=[(i,min(i+chunk,x.shape[1])) for i in range(0,x.shape[1],chunk)]
        else:
            bounds=offsets.cpu().tolist()
            spans=[(bounds[s]+c*chunk,min(bounds[s]+(c+1)*chunk,bounds[s+1])) for s,c in indices.cpu().tolist()]
        out=[]
        for a,b in spans:
            xx=x[:,a:b].permute(0,2,3,1).contiguous().reshape(-1,b-a)
            w=torch.ones(b-a,device=x.device,dtype=x.dtype)
            avg=fn(xx,w,None,1e-5)[1].to(x.dtype).reshape(x.shape[0],*x.shape[2:])
            out.append(avg)
        return torch.stack(out,dim=1) if out else x.new_empty((x.shape[0],0,*x.shape[2:]))
    return Candidate(Implementation(run),'sgl_kernel_npu.fla.layernorm_gated.layer_norm_fwd_npu_simd()[1]',
                     'Resolve exact supplied chunks each run; transpose each complete token-reduction axis into rows, select native mean without padding the divisor, restore chunk order.',
                     source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host),timing_mode='host_roundtrip')


def ada_zero(case):
    from dataclasses import replace
    from types import SimpleNamespace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    native=ada_layernorm(case)
    multiply=mul(case)
    def run(x,scale,shift,gate):
        # Exact widening preserves the reference's FP32 intermediate before gate.
        normalized=native.implementation.run(x.float(),scale.float(),shift.float())
        return multiply.implementation.run(normalized,gate.float()).to(x.dtype)
    return replace(native,implementation=Implementation(run),api=native.api+' + '+multiply.api,
                   adaptation='Lossless FP32 input/affine widening, original SGL adaptive LayerNorm groups, original SGL per-element gate multiplication, single final contract cast.',
                   source_sha256=native.source_sha256|multiply.source_sha256|source_hashes(Path(__file__)))


def minimum(case):
    from dataclasses import replace
    from types import SimpleNamespace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    from .elementwise import neg
    negative=neg(SimpleNamespace(inputs=(case.inputs[0],),params={}))
    native=maximum(case)
    def run(x,y):
        # Neg adapter shape is bound to x. Broadcast the pair first so x/y fit.
        import torch
        from .elementwise import sub
        shape=torch.broadcast_shapes(x.shape,y.shape)
        xx,yy=x.expand(shape).contiguous(),y.expand(shape).contiguous()
        n=sub(SimpleNamespace(inputs=(torch.zeros_like(xx),xx),params={'alpha':1}))
        nx=n.implementation.run(torch.zeros_like(xx),xx)
        ny=n.implementation.run(torch.zeros_like(yy),yy)
        selected=native.implementation.run(nx,ny)
        return n.implementation.run(torch.zeros_like(selected),selected)
    return replace(native,implementation=Implementation(run),api='sgl_kernel_npu::mul_add negative + '+native.api+' + native negative',
                   adaptation='Broadcast operands, native exact sign inversion, original SGL maximum selector, native inverse sign; all transformations and launches per call.',
                   source_sha256=native.source_sha256|negative.source_sha256|source_hashes(Path(__file__)))
