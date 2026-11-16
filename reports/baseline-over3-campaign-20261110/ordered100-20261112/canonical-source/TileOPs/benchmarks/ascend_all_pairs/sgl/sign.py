"""Finite-input sign from original first-tie argmax and native subtraction."""
from pathlib import Path


def build(case):
    import torch
    from types import SimpleNamespace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .comparison import build as compare
    from .extra_arithmetic import _affine
    x, = case.inputs
    spec = SimpleNamespace(inputs=(x, x), params={},
                           _call=SimpleNamespace(signature=SimpleNamespace(name='LtFwdOp')))
    less = compare(spec)
    affine, hashes = _affine(case)
    def run(x):
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        negative = less.implementation.run(x, zero).float()
        positive = less.implementation.run(zero, x).float()
        return affine(negative, positive, -1.0).to(x.dtype)
    return Candidate(Implementation(run), less.api+' + native mul_add sign difference',
        'Original strict first-tie selectors give x<0 and 0<x; native affine returns positive-minus-negative. Signed zero produces zero. No external comparison or sign arithmetic.',
        less.source_sha256 | hashes | source_hashes(Path(__file__)),
        limitations=('Scoped to finite floating values in the unchanged original manifest generators; no general NaN/integer semantics claimed.',))
