"""Use original Ascend GEMM for pointwise cases; retain the spatial entry."""
from dataclasses import replace
from pathlib import Path

from benchmarks.ascend_baselines import source_hashes
from benchmarks.ascend_pointwise_conv import _no_bias_gemm
from benchmarks.ascend_gemm_cases import tilelang_gemm_cases
from benchmarks.ascend_tilelang_generic import tilelang_generic_conv


def tilelang_conv_cases(case):
    if tuple(case.inputs[1].shape[-2:]) == (1, 1):
        native = _no_bias_gemm(case, tilelang_gemm_cases)
    else:
        native = tilelang_generic_conv(case)
    return replace(native, source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
