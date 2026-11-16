"""Symmetric INT8 dequantization through original FLA single-tap products."""
from pathlib import Path


def dequant(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .native_affine import make_affine
    affine,hashes=make_affine()
    name=case._call.signature.name
    dtype=getattr(torch,case.params['out_dtype'])

    def run(q,scale):
        if name=='INT8DequantPerChannelFwdOp':scale=scale[:,None]
        elif name=='INT8DequantPerBlockFwdOp':scale=scale.repeat_interleave(128,dim=1)[:,:q.shape[1]]
        return affine(q.float(),scale).to(dtype)

    return Candidate(Implementation(run),'FLA causal_conv1d_fwd_impl_old(W=1, FP32 tap=scale)',
        'INT8 values widen losslessly to FP32; repeat/broadcast original per-tensor, row or 128-column block scales with exact ragged crop. Original native product and final manifest output conversion; input-dependent copies remain inside run.',
        hashes|source_hashes(Path(__file__)))
