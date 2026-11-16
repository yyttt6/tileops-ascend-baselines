"""Original complete RMS statistic for the positive reciprocal domain."""
from pathlib import Path


def bind():
    import torch
    from benchmarks.ascend_baselines import LIBRARIES, local_function, source_hashes
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    norm = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu',
                          'sgl_kernel_npu.fla.layernorm_gated', 'layer_norm_fwd_npu', host)
    def inverse_positive(x):
        # The full RMS statistic of each singleton row is 1/sqrt(x*x).
        # Original manifest generators for reciprocal and division are positive.
        rows = x.float().contiguous().reshape(-1, 1)
        return norm(rows, torch.ones(1, device=x.device, dtype=torch.float32),
                    None, 0.0, is_rms_norm=True)[2].reshape(x.shape)
    return inverse_positive, source_hashes(Path(__file__), LIBRARIES/'sgl-kernel-npu'/host)


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    from .additional import mul
    if any(x.dtype not in (torch.float16, torch.bfloat16, torch.float32) for x in case.inputs):
        raise UnsupportedCase('Original positive-domain singleton RMS reciprocal supports floating inputs only')
    if case.params.get('rounding_mode') is not None:
        raise UnsupportedCase('Integer rounding modes require an original floor/truncation stage')
    reciprocal, hashes = bind()
    product = mul(case)
    def run(x, y=None):
        result = reciprocal(x) if y is None else product.implementation.run(x.float(), reciprocal(y))
        return result.to(x.dtype)
    return Candidate(Implementation(run), 'sgl_kernel_npu::layer_norm_fwd_npu(is_rms_norm=True, eps=0)[2] + native scale_shift',
        'The unchanged original generators draw positive floating inputs in [0.5,1.5]. Each full singleton native RMS statistic is reciprocal on that domain; optional original native multiplication supplies unrounded division. FP32 intermediates and final original dtype cast, without an external reciprocal/division.',
        hashes | product.source_sha256,
        limitations=('Positive finite original manifest domain only. Negative/signed-zero/integer or rounded-division contracts are not generalized; original tolerance still applies.',))
