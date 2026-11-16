"""Layout/API-only adapters for original SGL elementwise kernels."""
from pathlib import Path


def _gated(case, sigmoid_only):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/activation/fused_sigmoid_mul.py'
    fn = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu',
                        'sgl_kernel_npu.activation.fused_sigmoid_mul', 'fused_sigmoid_mul', host)
    def run(x):
        return fn(torch.ones_like(x) if sigmoid_only else x, x)
    return Candidate(Implementation(run), 'sgl_kernel_npu.activation.fused_sigmoid_mul.fused_sigmoid_mul',
                     'Original x*sigmoid(gate); constant-one x for sigmoid, shared input for SiLU. No input-dependent precomputation.',
                     source_hashes(Path(__file__), LIBRARIES / 'sgl-kernel-npu' / host,
                                   LIBRARIES / 'sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/utils/triton_utils.py'))


def sigmoid(case):
    return _gated(case, True)


def silu(case):
    return _gated(case, False)
