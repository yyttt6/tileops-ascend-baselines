"""Complete original SGEMMV reduction rows with exact constant coefficients."""
from pathlib import Path


def build(case, *, average):
    import importlib
    import sys
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,UnsupportedCase,_owned_native,source_hashes
    from .reduction import row_layout
    pack,width,shape=row_layout(case)
    x=case.inputs[0]
    if x.dtype not in (torch.float16,torch.bfloat16):
        raise UnsupportedCase('Original SGL SGEMMV accepts FP16/BF16 inputs without lossy conversion')
    if average and width & (width-1):
        raise UnsupportedCase('Mean GEMV requires a power-of-two row for an exactly represented reciprocal coefficient')
    root=LIBRARIES/'sgl-kernel-npu'
    package=root/'python/sgl_kernel_npu'
    sys.path.insert(0,str(package))
    module=importlib.import_module('sgl_kernel_npu')
    if Path(module.__file__).resolve()!=package/'sgl_kernel_npu/__init__.py':raise RuntimeError('SGL source mismatch')
    binary=package/'sgl_kernel_npu/lib/libsgl_kernel_npu.so'
    _owned_native(binary,['_ZN6sglang10npu_kernel13sgemmv_shrinkERN2at6TensorES3_S3_S3_S3_S3_S3_l'])
    rows=x.numel()//width
    padded=max(32,(width+15)//16*16)
    weights=F.pad(torch.full((1,16,width),1.0/width if average else 1.0,device=x.device,dtype=x.dtype),(0,padded-width))
    indices=torch.zeros(1,device=x.device,dtype=torch.int32)
    lengths=torch.tensor([rows],device=x.device,dtype=torch.int32)
    ranks=torch.tensor([16],device=x.device,dtype=torch.int32)
    scales=torch.ones(1,device=x.device,dtype=torch.float16)
    output=torch.empty((rows,16),device=x.device,dtype=torch.float32)
    dtype=getattr(torch,case.params.get('dtype') or str(x.dtype).split('.')[-1])
    def run(x):
        matrix=F.pad(pack(x),(0,padded-width))
        torch.ops.npu.sgemmv_shrink(matrix,weights,indices,lengths,ranks,scales,output,1)
        return output[:,0].to(dtype).reshape(shape).contiguous()
    sources=[p for p in (root/'csrc/lora').rglob('*') if p.suffix in ('.cpp','.h','.hpp')]
    return Candidate(Implementation(run),'sgl_kernel_npu::sgemmv_shrink (complete-row '+('mean' if average else 'sum')+')',
        'Pack the full original reduction axis into native GEMV rows; exact constant weights are repeated across sixteen independent native output columns for alignment, zero-pad K, select one equal column and cast once to declared output dtype. Original FP32 accumulator, kernel schedule and complete K reduction unchanged; no partial reductions or external reduction arithmetic.',
        source_hashes(Path(__file__),Path(__file__).with_name('reduction.py'),binary,package/'sgl_kernel_npu/__init__.py',root/'csrc/pytorch_extensions.cpp',*sources))


def mean(case):
    import torch
    from .reduction import row_layout
    width=row_layout(case)[1]
    if case.inputs[0].dtype in (torch.float16,torch.bfloat16) and width>16384 and not width & (width-1):
        return build(case,average=True)
    from .mean_compat import mean as original
    return original(case)


def sum_reduce(case):
    import torch
    if case.inputs[0].dtype in (torch.float16,torch.bfloat16):return build(case,average=False)
    from .reduction import row_layout
    pack,width,shape=row_layout(case)
    if case.inputs[0].dtype == torch.float32 and width>8192:
        from dataclasses import replace
        from benchmarks.api import Implementation
        from benchmarks.ascend_baselines import source_hashes
        from .mean_compat import mean as complete_mean
        from .extra_arithmetic import _affine
        native=complete_mean(case)
        affine,hashes=_affine(case)
        def run(x):
            value=native.implementation.run(x)
            return affine(value,torch.zeros_like(value),width).reshape(shape)
        return replace(native,implementation=Implementation(run),
                       api=native.api+' + original native mul_add width scaling',
                       adaptation='Existing alternate complete-row mean API avoids the already observed SIMD mean UB overflow; original native width scaling restores sum. No partial reduction or tolerance change.',
                       source_sha256=native.source_sha256|hashes|source_hashes(Path(__file__)))
    from .scalar import sum_reduce as original
    return original(case)
