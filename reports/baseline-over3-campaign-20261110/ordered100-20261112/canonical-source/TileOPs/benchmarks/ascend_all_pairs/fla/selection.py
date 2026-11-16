"""Finite floating selection through unchanged original FLA tap arithmetic."""
from pathlib import Path


def selection(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .native_affine import make_affine
    affine,hashes=make_affine()
    name=case._call.signature.name

    def run(*args):
        if name=='WhereFwdOp':
            mask,left,right=args
        else:
            right,mask,*value=args
            left=value[0] if value else torch.full((),case.params['value'],device=right.device,dtype=right.dtype)
        shape=torch.broadcast_shapes(mask.shape,left.shape,right.shape)
        dtype=right.dtype
        mask=mask.float().expand(shape)
        inverse=affine(mask,-1.0,torch.ones((),device=right.device,dtype=torch.float32))
        other=affine(right.float().expand(shape),inverse)
        return affine(left.float().expand(shape),mask,other).to(dtype)

    return Candidate(Implementation(run),'FLA original causal FP32 taps with lossless boolean 0/1 weights',
        'For finite floating inputs, select by mask*x + (1-mask)*y using original FLA native multiply/add. Bool->FP32 is exact, multiplication by 0/1 and adding zero preserve the selected finite value; broadcast/copy/final cast and current scalar value all handled inside run. Nonfinite inputs are not covered by this arithmetic identity and remain outside the finite manifest-generated domain.',
        hashes|source_hashes(Path(__file__)))
