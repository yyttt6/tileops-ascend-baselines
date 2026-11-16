"""INT8 GEMM through lossless widening and original FLA matrix arithmetic."""
from pathlib import Path


def int8_matmul(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .cube_matmul import make_matmul
    from .native_affine import make_affine
    matmul, hashes = make_matmul(torch.float32)
    affine, affine_hashes = make_affine()
    output_dtype = getattr(torch, case.params['out_dtype'])

    def run(a, b, scale_a, scale_b, bias=None):
        # Every signed INT8 value is exactly representable in BF16. Keep the
        # native accumulated output in FP32 until both current scales and bias
        # have been applied; never round scaled operands back to BF16.
        value = matmul(a.to(torch.bfloat16), b.to(torch.bfloat16).t().contiguous())
        value = affine(affine(value, scale_a), scale_b)
        if bias is not None:
            value = affine(value, 1.0, bias.float())
        return value.to(output_dtype)

    return Candidate(
        Implementation(run),
        'FLA ChunkFwdH final FP32 state + native FP32 scale/residual, lossless INT8-to-BF16 widening',
        'Widen original INT8 values exactly. Original FLA state recurrence computes the matrix product; current tensor/row/column scales and optional bias use original tap/residual kernels. FP32 accumulation is checked against the unchanged exact INT32 manifest oracle and tolerance. All transformations occur inside run; cast only the final result.',
        hashes | affine_hashes | source_hashes(Path(__file__)),
    )
