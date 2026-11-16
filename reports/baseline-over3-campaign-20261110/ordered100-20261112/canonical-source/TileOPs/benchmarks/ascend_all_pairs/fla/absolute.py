"""Absolute values from the original scalar L2 reciprocal-norm contract."""
from pathlib import Path


def absolute(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .providers import _triton_module
    norm,source=_triton_module('l2norm')

    def run(x):
        pieces=[]
        flat=x.contiguous().reshape(-1)
        # Bound the two original FP32 reciprocal outputs without altering the
        # native kernel's own launch dimensions or schedule.
        for start in range(0,flat.numel(),1048576):
            values=flat[start:start+1048576].reshape(-1,1)
            _,inverse=norm.l2norm_fwd(values,eps=0.0,output_dtype=torch.float32)
            _,magnitude=norm.l2norm_fwd(inverse.reshape(-1,1),eps=0.0,output_dtype=torch.float32)
            pieces.append(magnitude.to(x.dtype))
        return torch.cat(pieces).reshape(x.shape)

    return Candidate(Implementation(run),'FLA scalar l2norm_fwd(rstd), twice',
        'Reshape every element as an independent scalar L2 row; two original reciprocal-norm outputs recover its magnitude. Zero and infinite rows follow reciprocal composition. Finite FP32 extremes can overflow/underflow the original squared intermediate and remain subject to original numerical checks. All packing inside run; no torch value arithmetic.',
        source_hashes(Path(__file__),source,source.with_name('utils.py')))
