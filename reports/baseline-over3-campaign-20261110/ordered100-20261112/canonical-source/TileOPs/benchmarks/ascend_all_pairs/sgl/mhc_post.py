"""MHC post projection from complete original native product and affine APIs."""
from pathlib import Path


def build(case):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .additional import mul
    from .extra_arithmetic import _affine
    product = mul(case)
    affine, hashes = _affine(case)

    def run(x_layer_out, h_post, x_res):
        # The reference's matrix product has reduction width one: an outer
        # product of each supplied expansion coefficient and complete feature row.
        expanded = product.implementation.run(
            x_layer_out.float()[:, None, :], h_post.float()[:, :, None])
        return affine(expanded.reshape_as(x_res), x_res.float(), 1.0).to(x_res.dtype)

    return Candidate(Implementation(run),
        'sgl_kernel_npu::fused_scale_shift(scale_constant=0) + mul_add',
        'Broadcast the supplied per-row expansion coefficients against the full feature row; original native FP32 multiplication and original native addition reproduce the complete width-one outer-product plus residual contract, followed by its declared BF16 cast. No external tensor arithmetic or omitted outputs.',
        product.source_sha256 | hashes | source_hashes(Path(__file__)))
