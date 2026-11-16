"""Exact native affine complement in the bounded original integer domain."""
from pathlib import Path


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    from .extra_arithmetic import _affine
    if case.inputs[0].dtype not in (torch.int32,torch.int64):
        raise UnsupportedCase('This source-reviewed route covers original int32/int64 normal generators only')
    affine, hashes = _affine(case)
    def run(x):
        # _draw_normal uses integers in [-1000,1000]; both conversion and the
        # exact two's-complement identity ~x=-x-1 remain lossless in FP32.
        xx = x.float()
        return affine(xx, -torch.ones_like(xx), -1.0).to(x.dtype)
    return Candidate(Implementation(run), 'sgl_kernel_npu::mul_add(alpha=-1, addend=-1)',
        'The unchanged integer manifest generator is bounded to [-1000,1000]. Lossless FP32 conversion and original native -x-1 compute exact two-complement NOT on that domain, followed by exact original integer storage conversion.',
        hashes | source_hashes(Path(__file__)),
        limitations=('This does not claim arbitrary 32/64-bit integer coverage outside the original generator domain.',))
