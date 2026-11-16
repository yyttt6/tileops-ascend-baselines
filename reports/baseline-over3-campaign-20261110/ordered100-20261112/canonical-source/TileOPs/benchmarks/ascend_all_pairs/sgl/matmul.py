"""Batch independent matrix products using SGL's own vector LoRA entry."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def bmm(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    from benchmarks.ascend_sgl_gemv_cases import sgl_gemv_cases
    a,b = case.inputs
    natives = [sgl_gemv_cases(SimpleNamespace(inputs=(a[i],b[i]),params={'trans_a':False,'trans_b':False})) for i in range(a.shape[0])]
    def run(a,b):
        return torch.stack([native.implementation.run(a[i],b[i]) for i,native in enumerate(natives)])
    hashes = source_hashes(Path(__file__))
    for native in natives: hashes.update(native.source_sha256)
    return replace(natives[0],implementation=Implementation(run),source_sha256=hashes,
                   adaptation='Independent original GEMM per batch, complete K reduction, stack batch outputs; every transpose/pad/launch retained in run.')


def grouped_gemm(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    from benchmarks.ascend_sgl_gemv_cases import sgl_gemv_cases
    a,b,sizes,offsets = case.inputs
    ta,tb = case.params['transpose_a'],case.params['transpose_b']
    # Factories capture shapes only. Metadata is read again in every invocation.
    def run(a,b,sizes,offsets):
        spans = list(zip(offsets.cpu().tolist(),sizes.cpu().tolist()))
        outputs=[]
        for i,(begin,length) in enumerate(spans):
            aa=a[begin:begin+length]
            bb=(b[:,begin:begin+length] if tb else b[begin:begin+length]) if ta else b[i]
            if length == 0:
                n = b.shape[0] if (ta and tb) else b.shape[-1] if ta else b.shape[-2] if tb else b.shape[-1]
                outputs.append(torch.zeros((a.shape[1],n) if ta else (0,n),device=a.device,dtype=a.dtype))
                continue
            native=sgl_gemv_cases(SimpleNamespace(inputs=(aa,bb),params={'trans_a':ta,'trans_b':tb}))
            outputs.append(native.implementation.run(aa,bb))
        return torch.stack(outputs) if ta else torch.cat(outputs,dim=0)
    # Fingerprint all source paths without depending on a nonempty first group.
    from benchmarks.ascend_baselines import Candidate,LIBRARIES
    src=LIBRARIES/'sgl-kernel-npu'
    files=[f for f in (src/'csrc/lora').rglob('*') if f.suffix in ('.cpp','.h','.hpp')]
    return Candidate(Implementation(run),'sgl_kernel_npu::sgemmv_shrink grouped independent products',
                     'Read dynamic group offsets/sizes on each call; complete per-group K reductions and stack/concatenate; empty groups contain no products. Host metadata transfers included.',
                     source_hashes(Path(__file__),Path(__file__).parents[2]/'ascend_sgl_gemv_cases.py',Path(__file__).parents[2]/'ascend_sgl_gemv_pilot.py',src/'csrc/pytorch_extensions.cpp',src/'python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so',*files),
                     timing_mode='host_roundtrip')
