"""Native distance and slope products for the complete ALiBi tensor."""
from pathlib import Path


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from workloads.device import run_device
    from .additional import maximum, mul
    from .extra_arithmetic import _affine
    affine, hashes = _affine(case)
    select, product = maximum(case), mul(case)
    n, heads = case.params['seq_len'], case.params['num_heads']
    dtype = getattr(torch, case.params['out_dtype'])
    # These public coefficients depend only on shape, like the original norm
    # scale and correction constants. No tensor distance/output is cached.
    coefficients = [-2.0**(-8.0*(h+1)/heads) for h in range(heads)]
    def run():
        device = case.params.get('device') or run_device()
        position = torch.arange(n, device=device, dtype=torch.float32)
        delta = affine(position[:, None], position[None, :], -1.0)
        negative = affine(delta, torch.zeros_like(delta), -1.0)
        distance = select.implementation.run(delta, negative)
        slopes = torch.tensor(coefficients, device=device, dtype=torch.float32).reshape(heads,1,1)
        return product.implementation.run(distance.unsqueeze(0), slopes).to(dtype)
    return Candidate(Implementation(run), 'sgl_kernel_npu::mul_add distance + native argmax absolute value + scale_shift slope',
        'Generate position coordinates and public shape-dependent slope constants, then original native subtraction, absolute-value selector and slope multiplication return the full [heads,sequence,sequence] ALiBi bias. No external distance tensor arithmetic or precomputed output.',
        hashes | select.source_sha256 | product.source_sha256 | source_hashes(Path(__file__)))
