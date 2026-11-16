"""Complete native row reciprocal standard deviation, then native reciprocal."""
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
    correction = case.params.get('correction')
    correction = 1 if correction is None else correction
    if width <= correction:
        raise UnsupportedCase('Nonpositive standard deviation degrees of freedom are not supplied by this route')
    if width*case.inputs[0].element_size() > 65536:
        raise UnsupportedCase('Original complete LayerNorm public API rejects a feature row larger than 64KiB')
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    entry = 'layer_norm_fwd_npu' if width>8192 else 'layer_norm_fwd_npu_simd'
    norm = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu', 'sgl_kernel_npu.fla.layernorm_gated', entry, host)
    reciprocal, hashes = bind()
    affine, affine_hashes = _affine(case)
    factor = math.sqrt(width/(width-correction))
    def run(x):
        rows = pack(x)
        rstd = norm(rows, torch.ones(width, device=x.device, dtype=x.dtype), None, 0.0)[2]
        std = reciprocal(rstd)
        return affine(std, torch.zeros_like(std), factor).to(x.dtype).reshape(shape)
    return Candidate(Implementation(run), 'sgl_kernel_npu::complete LayerNorm rstd + singleton RMS rstd + native correction',
        'Select the complete original native reciprocal population standard deviation with eps=0, apply a complete singleton RMS reciprocal statistic, and original native scalar sqrt(N/(N-correction)) correction. The factor depends only on shape/contract metadata. Every original reduction term and output shape is retained.',
        hashes | affine_hashes | source_hashes(Path(__file__),Path(__file__).with_name('reduction.py'), LIBRARIES/'sgl-kernel-npu'/host),
        limitations=('Full native row resource limits and rounding remain subject to unchanged original per-case validation.',))
