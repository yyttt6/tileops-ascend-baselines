"""Exact low-precision product through the original MhcPost no-HRes API."""
from pathlib import Path


def multiply(case):
    import torch
    from dataclasses import replace
    from .native import GraphCall
    from .adapters import _candidate
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
    from .hashing import source_hashes
    x,y=case.inputs
    if x.dtype not in (torch.float16,torch.bfloat16) or y.dtype!=x.dtype:
        raise UnsupportedCase('MhcPost h_out/x pointers require matching FP16 or BF16; FP32 inputs cannot be narrowed losslessly.')
    binding=GraphCall('ops-transformer')
    def run(x,y):
        a,b=torch.broadcast_tensors(x,y);shape=a.shape
        # T independently batched scalar products, n=D=1. Original API does
        # FP32 h_post * widened h_out + zero residual, then casts once to x dtype.
        post=a.contiguous().reshape(-1,1).float();layer=b.contiguous().reshape(-1,1)
        residual=torch.zeros((post.shape[0],1,1),dtype=x.dtype,device=x.device)
        out=torch.empty_like(residual)
        binding.call('MhcPost',[residual,None,layer,post],[out],{})
        return out.reshape(shape)
    result=_candidate(binding,run,'MhcPost','mhc/mhc_post','Broadcast and flatten scalar products as independent T batches with n=D=1; original no-HRes MhcPost computes FP32 h_post*h_out plus a zero residual and casts to the manifest input dtype. Only exact widening/layout/constants outside the native API; all work is timed.')
    return replace(result,source_sha256=result.source_sha256|source_hashes(Path(__file__)))
