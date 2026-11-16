"""Declared tanh-GELU formula through complete original SGL pointwise APIs."""
from pathlib import Path


def build(case):
    import math
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes
    from .additional import mul
    from .extra_arithmetic import _affine
    product = mul(case)
    affine, hashes = _affine(case)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/activation/fused_sigmoid_mul.py'
    sigmoid_mul = local_function('sgl-kernel-npu','python/sgl_kernel_npu',
        'sgl_kernel_npu.activation.fused_sigmoid_mul','fused_sigmoid_mul',host)
    def run(x):
        gate,up = (v.float() for v in x.chunk(2,dim=-1))
        square = product.implementation.run(gate,gate)
        cube = product.implementation.run(square,gate)
        polynomial = affine(cube,gate,0.044715)
        doubled = affine(polynomial,torch.zeros_like(polynomial),2*math.sqrt(2/math.pi))
        # 0.5*x*(1+tanh(z)) == x*sigmoid(2*z), without tail cancellation.
        activated = sigmoid_mul(gate,doubled)
        return product.implementation.run(activated,up).to(x.dtype)
    return Candidate(Implementation(run),'sgl_kernel_npu::scale_shift/mul_add + fused_sigmoid_mul + native product',
        'Only the explicitly declared tanh-GELU variant: original native products form x cubed, native affine uses the public 0.044715 and sqrt(2/pi) constants, and the complete native x*sigmoid(2*z) identity supplies tanh-GELU before native up multiplication. FP32 intermediates match the original fused workload; final original dtype cast once.',
        product.source_sha256 | hashes | source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host),
        limitations=('Does not replace exact-erf GELU. Tanh/sigmoid identity rounding remains subject to the unchanged fused-gated tolerance.',))
