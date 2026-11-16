"""Original grouped FP32 accumulator followed by owned complete SiLU gate.

The original full-output GroupedMatmulAdd and fused_silu components require
native receipts. No rounding of the grouped result before activation.
"""
from pathlib import Path


def grouped_gated(case):
    import torch
    from dataclasses import replace
    from types import SimpleNamespace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import UnsupportedCase
    from .adapters import grouped_gemm
    from .fused_silu import fused_silu
    from .hashing import source_hashes

    if case._call.signature.name != 'MoEGroupedGemmFwdOp':
        raise UnsupportedCase('This binding implements original grouped-expert GEMM only.')
    if case.params.get('activation') != 'silu_and_mul':
        raise UnsupportedCase('Only the original native sigmoid/product gate is bound.')
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16):
        raise UnsupportedCase('Original GroupedMatmulAdd requires lossless FP16/BF16 input.')
    output_dtype = getattr(torch, case.params['out_dtype']) if case.params.get('out_dtype') else case.inputs[0].dtype
    # Override only adapter routing metadata. The original workload/reference
    # and case are unchanged, and all matrix products remain in native run().
    matrix_case = SimpleNamespace(_call=case._call, inputs=case.inputs,
                                  params=case.params | {'activation': None, 'out_dtype': 'float32'})
    matrix = grouped_gemm(matrix_case)
    gate = fused_silu(case)

    def run(*inputs):
        accumulator = matrix.implementation.run(*inputs)
        if accumulator.dtype != torch.float32:
            raise RuntimeError('Grouped native accumulator was rounded before the gate.')
        return gate.implementation.run(accumulator).to(output_dtype)

    def cleanup():
        if matrix.cleanup is not None:
            matrix.cleanup()
        if gate.cleanup is not None:
            gate.cleanup()

    return replace(matrix, implementation=Implementation(run),
                   api='GroupedMatmulAdd+MoeGatingTopK+MoeFinalizeRoutingV2',
                   adaptation='Original grouped expert layouts and complete K GEMM retain FP32 accumulators. '
                              'Original full sigmoid plus two original FP32 routing products compute SiLU(gate)*up. '
                              'Only the complete final result is cast to manifest dtype; no Torch numerical fallback, '
                              'kernel edit, tiling override, or input-dependent build copy.',
                   source_sha256=matrix.source_sha256 | gate.source_sha256 | source_hashes(Path(__file__)),
                   cleanup=cleanup)
