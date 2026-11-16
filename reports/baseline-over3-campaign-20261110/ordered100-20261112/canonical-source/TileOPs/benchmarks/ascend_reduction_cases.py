"""Pack manifest reduction axes for existing original row-max candidates."""
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

from benchmarks.ascend_baselines import source_hashes


def row_max_case(case, factory, row_multiple=1):
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    x, = case.inputs
    dims = case.params['dim']
    dims = tuple(range(x.ndim)) if dims is None else ((dims,) if isinstance(dims, int) else tuple(dims))
    axes = tuple(d % x.ndim for d in dims)
    kept = tuple(d for d in range(x.ndim) if d not in axes)
    width = math.prod(x.shape[d] for d in axes)
    rows = x.numel() // width
    shape = (tuple(1 if d in axes else n for d, n in enumerate(x.shape))
             if case.params['keepdim'] else tuple(x.shape[d] for d in kept))
    def pack(x):
        value = x.permute(*(kept + axes)).contiguous().reshape(rows, width)
        return F.pad(value, (0, 0, 0, -rows % row_multiple)) if rows % row_multiple else value
    native = factory(SimpleNamespace(inputs=(pack(x),), params=case.params | {'dim': -1, 'keepdim': False}))
    def run(x):
        return native.implementation.run(pack(x)).reshape(-1)[:rows].reshape(shape)
    return replace(native, implementation=Implementation(run),
                   adaptation='Permute and flatten reduction axes into complete rows, pad independent rows only where needed; original row maximum; restore manifest output shape.',
                   source_sha256=native.source_sha256 | source_hashes(Path(__file__)))


def tilekernels_amax_cases(case):
    from benchmarks.ascend_selection_baselines import tilekernels_amax
    return row_max_case(case, tilekernels_amax)


def sgl_amax_cases(case):
    from benchmarks.ascend_sgl_more import sgl_amax
    return row_max_case(case, sgl_amax)


def tla_amax_cases(case):
    from benchmarks.ascend_tla_reduction import tla_amax
    return row_max_case(case, tla_amax, row_multiple=4)


def opsmath_amax_cases(case):
    from benchmarks.ascend_math_graph import opsmath_graph_amax
    return row_max_case(case, opsmath_graph_amax)


def transformer_amax_cases(case):
    from benchmarks.ascend_transformer_graph import transformer_graph_amax
    return row_max_case(case, transformer_graph_amax)


def vllm_amax_cases(case):
    from benchmarks.ascend_vllm_amax import vllm_amax
    return row_max_case(case, vllm_amax)
