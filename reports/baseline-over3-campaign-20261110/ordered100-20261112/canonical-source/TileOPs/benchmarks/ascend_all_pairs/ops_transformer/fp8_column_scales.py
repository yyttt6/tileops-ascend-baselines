"""Staged complete GEMM using original G-B products for independent columns.

An N=1 native product has exactly one N128 scale block. Its scale is the
original weight row's K128 grid. Every output column retains the complete M/K
extent; no scales are averaged, no K reduction is split, and no operand is
dequantized outside the original kernel. Unregistered pending native evidence.
"""
from pathlib import Path


def fp8_column_scales(case):
    import torch
    from dataclasses import replace
    from benchmarks.ascend_baselines import UnsupportedCase
    from .native import GraphCall
    from .adapters import _candidate, _fp8_block_product
    from .hashing import source_hashes

    if case._call.signature.name != 'GemmFP8FwdOp' or len(case.inputs) != 4:
        raise UnsupportedCase('Only original bias-free per-row FP8 GEMM is bound.')
    a, b, sa, sb = case.inputs
    m, k = a.shape
    n = b.shape[0]
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype or b.shape[1] != k:
        raise UnsupportedCase('Original matching E4M3 operands and full K are required.')
    if k % 128 or sa.shape != (m, k // 128) or sb.shape != (n, k // 128):
        raise UnsupportedCase('Original independent row/K128 scale grids are required.')
    if sa.dtype != torch.float32 or sb.dtype != torch.float32:
        raise UnsupportedCase('Original FP32 scales are preserved without conversion.')
    dtype = getattr(torch, case.params['out_dtype'])
    binding = GraphCall('ops-transformer')

    def run(a, b, sa, sb):
        output = torch.empty((a.shape[0], b.shape[0]), dtype=dtype, device=a.device)
        for column in range(b.shape[0]):
            value = _fp8_block_product(binding, a, b[column:column + 1], sa,
                                       sb[column:column + 1], dtype)
            output[:, column:column + 1].copy_(value)
            binding.cleanup()
        return output

    result = _candidate(binding, run, 'GroupedMatmul', 'gmm/grouped_matmul',
                        'Complete original MxN output from one original G-B GEMM per independent '
                        'weight row, each retaining complete M and K. N=1 maps its original '
                        'per-row/K128 weight scale to one exact N128/K128 native scale block. '
                        'No K splitting, scale averaging, external dequantization, arithmetic '
                        'fallback, native tiling override or tolerance change; full original '
                        'case comparison is mandatory. Many launches may be slow.', host=True)
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__)))
