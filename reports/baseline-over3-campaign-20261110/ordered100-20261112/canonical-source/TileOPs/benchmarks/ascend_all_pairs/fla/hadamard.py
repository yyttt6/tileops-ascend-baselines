"""Hadamard transforms composed entirely from original FLA tap/residual APIs."""
from math import sqrt
from pathlib import Path


def hadamard(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from workloads import ascend_extended_transform_quant as convention
    from .native_affine import make_affine
    affine,hashes=make_affine()
    base=case.params['base_order']
    # These are the manifest's fixed mathematical sign tables, not an oracle
    # computation. No input-dependent arithmetic runs in this module on CPU.
    signs=None if base==1 else convention._HADAMARD_SIGNS[base]

    def run(x):
        n=x.shape[-1];radix=n//base
        values=x.contiguous().float().reshape(-1,base,radix)
        width=1
        while width<radix:
            pairs=values.reshape(values.shape[0],base,radix//(2*width),2,width)
            left=pairs[...,0,:].contiguous();right=pairs[...,1,:].contiguous()
            summed=affine(left,1.0,right)
            differenced=affine(right,-1.0,left)
            values=torch.stack((summed,differenced),dim=-2).reshape(values.shape)
            width*=2
        if signs is not None:
            outputs=[]
            for row in signs:
                accumulator=None
                for column,sign in enumerate(row):
                    accumulator=affine(values[:,column],1.0 if sign=='+' else -1.0,accumulator)
                outputs.append(accumulator)
            values=torch.stack(outputs,dim=1)
        return affine(values,1.0/sqrt(n)).to(x.dtype).reshape(x.shape)

    return Candidate(Implementation(run),'FLA causal_conv1d_fwd_impl_old(tap/residual Hadamard composition)',
        'Copy-only radix-two pair packing; every sum, difference, fixed base-order sign-matrix product and final normalization is an original FLA tap/residual call in FP32. Base-order conventions are the manifest mathematical constants, including order20; no zero-padding substitution. Final output cast once.',
        hashes|source_hashes(Path(__file__),Path(convention.__file__)))
