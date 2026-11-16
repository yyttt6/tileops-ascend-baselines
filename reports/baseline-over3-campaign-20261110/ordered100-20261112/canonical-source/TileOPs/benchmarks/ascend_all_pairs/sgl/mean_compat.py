"""Select the vendor's second public complete-row implementation when needed."""
from pathlib import Path


def mean(case):
    import torch
    from .reduction import row_layout,mean as direct
    pack,width,shape=row_layout(case)
    x=case.inputs[0]
    if not (width>16384 or x.dtype==torch.float32 and width>8192):
        return direct(case)
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,local_function,source_hashes,LIBRARIES
    host='python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu',host)
    dtype=getattr(torch,case.params.get('dtype') or str(x.dtype).split('.')[-1])
    def run(x):
        return fn(pack(x),torch.ones(width,device=x.device,dtype=x.dtype),None,1e-5)[1].to(dtype).reshape(shape)
    return Candidate(Implementation(run),'sgl_kernel_npu.fla.layernorm_gated.layer_norm_fwd_npu()[1]',
        'Select the existing alternate public complete-row LayerNorm API after SIMD-entry UB overflow. Preserve native BLOCK_M=48 and all native schedule/math unchanged; select its FP32 mean output.',
        source_hashes(Path(__file__),Path(__file__).with_name('reduction.py'),LIBRARIES/'sgl-kernel-npu'/host))
