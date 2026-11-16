"""Shape and exact-cast wrappers for existing RMSNorm candidates."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from benchmarks.ascend_baselines import UnsupportedCase, _rms_layout, source_hashes


def _adapt(case, factory, *, float_inputs=False, row_multiple=1, max_hidden=None):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    x, shape, hidden, eps, pack = _rms_layout(case)
    if max_hidden is not None and hidden > max_hidden:
        raise UnsupportedCase('Original native reduction limit cannot be changed by independent-row padding.')
    rows = x.numel() // hidden
    padded_rows = (rows + row_multiple - 1) // row_multiple * row_multiple
    def prepare(x, weight):
        packed, gamma = pack(x, weight)
        if float_inputs:
            packed, gamma = packed.to(torch.float32), gamma.to(torch.float32)
        if padded_rows != rows:
            packed = F.pad(packed, (0, 0, 0, padded_rows - rows))
        return packed, gamma
    native_case = SimpleNamespace(inputs=prepare(*case.inputs),
                                  params=case.params | {'eps': eps, 'normalized_shape': [hidden]})
    native = factory(native_case)
    def run(x, weight):
        out = native.implementation.run(*prepare(x, weight))
        return out[:rows].reshape(shape).to(x.dtype)
    return replace(native, implementation=Implementation(run),
                   adaptation='Pack independent rows; preserve complete normalization axis; original manifest epsilon/unit-weight defaults; exact FP32 widening or independent-row zero padding where required; restore output shape/dtype.',
                   source_sha256=native.source_sha256 | source_hashes(Path(__file__)))


def tilekernels_rms_cases(case):
    import torch
    from benchmarks.ascend_baselines import tilekernels_rms
    x = case.inputs[0]
    # FP16 is exact in FP32. BF16 wider than the native BF16 lane-index limit
    # is also exact in the original FP32 template, without splitting a reduction.
    widen = x.dtype == torch.float16 or x.shape[-1] > 32768
    return _adapt(case, tilekernels_rms, float_inputs=widen)


def tilelang_rms_cases(case):
    from benchmarks.ascend_tilelang_baselines import tilelang_rms
    return _adapt(case, tilelang_rms, row_multiple=64)


def opsnn_rms_cases(case):
    from .ascend_nn_direct import opsnn_direct_rms
    return _adapt(case, opsnn_direct_rms)


def transformer_rms_cases(case):
    from benchmarks.ascend_transformer_graph import transformer_graph_rms
    return _adapt(case, transformer_graph_rms, max_hidden=1024)
