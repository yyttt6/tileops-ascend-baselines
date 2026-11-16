"""Select the complete Softplus branch in the original GDN gating API."""
from pathlib import Path


def build(case):
    import torch
    from types import SimpleNamespace
    from dataclasses import replace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import local_function, source_hashes, LIBRARIES
    from .elementwise import neg
    host='python/sgl_kernel_npu/sgl_kernel_npu/fla/fused_gdn_gating.py'
    gating=local_function('sgl-kernel-npu','python/sgl_kernel_npu',
                         'sgl_kernel_npu.fla.fused_gdn_gating','fused_gdn_gating_npu',host)
    x,=case.inputs
    negative=neg(SimpleNamespace(inputs=(x.float(),),params={}))
    def run(x):
        # exp(A_log)=1, dt_bias=0 makes the entire original returned g equal
        # -softplus(x; beta, threshold). Fixed-width independent element rows
        # only change layout, while preserving the vendor launch schedule.
        packed=x.contiguous().reshape(-1,8)
        zero=torch.zeros(8,device=x.device,dtype=torch.float32)
        g,_=gating(zero,packed,torch.zeros_like(packed),zero,
                   beta=case.params['beta'],threshold=case.params['threshold'])
        return negative.implementation.run(g.reshape_as(x)).to(x.dtype)
    return replace(negative,implementation=Implementation(run),
                   api='sgl_kernel_npu.fla.fused_gdn_gating.fused_gdn_gating_npu + original mul_add sign inversion',
                   adaptation='Original full Softplus branch with A_log=0 and dt_bias=0, original beta/threshold, independent 8-column element layout, native sign inversion and final output storage cast. All input transformations are per-call.',
                   source_sha256=negative.source_sha256|source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))
