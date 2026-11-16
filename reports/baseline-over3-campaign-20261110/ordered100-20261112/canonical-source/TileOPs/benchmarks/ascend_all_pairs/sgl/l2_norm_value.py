"""Complete native RMS statistic plus native reciprocal yields an L2 value."""
from pathlib import Path


def build(case):
    import math
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes
    from .reduction import row_layout
    from .inverse_rms import bind
    from .extra_arithmetic import _affine
    pack, width, shape = row_layout(case)
    x = case.inputs[0]
    if width*x.element_size()>65536:
        raise UnsupportedCase('Original complete RMS public API rejects feature rows larger than 64KiB')
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    entry = 'layer_norm_fwd_npu' if width>8192 else 'layer_norm_fwd_npu_simd'
    norm = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu', 'sgl_kernel_npu.fla.layernorm_gated', entry, host)
    reciprocal, hashes = bind()
    affine, affine_hashes = _affine(case)
    dtype = getattr(torch,case.params.get('dtype') or str(x.dtype).split('.')[-1])
    factor = math.sqrt(width)
    def run(x):
        rows = pack(x)
        rstd = norm(rows, torch.ones(width,device=x.device,dtype=x.dtype), None, 0.0, is_rms_norm=True)[2]
        rms = reciprocal(rstd)
        return affine(rms, torch.zeros_like(rms), factor).to(dtype).reshape(shape)
    return Candidate(Implementation(run), 'sgl_kernel_npu::complete RMS rstd + singleton RMS reciprocal + native sqrt(width) scaling',
        'The complete original RMS statistic is reciprocal RMS; its original native singleton-RMS reciprocal gives RMS, and original native sqrt(width) metadata scaling gives L2. Full reduction rows, eps=0, FP32 statistics and one declared output conversion; no external norm or tensor square root.',
        hashes | affine_hashes | source_hashes(Path(__file__), Path(__file__).with_name('reduction.py'),LIBRARIES/'sgl-kernel-npu'/host))
