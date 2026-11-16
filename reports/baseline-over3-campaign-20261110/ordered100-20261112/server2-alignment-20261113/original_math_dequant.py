"""Symmetric INT8 dequantization is the original Mul with scale layout."""
from pathlib import Path
from types import SimpleNamespace

def make(case,op):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .graph import make as graph_make
    p=case.params;shape=p['x'][0]
    child=graph_make(SimpleNamespace(params={'output':(shape,'float32')}),'MulFwdOp')
    def run(q,scale):
        if op=='INT8DequantPerChannelFwdOp':scale=scale[:,None]
        elif op=='INT8DequantPerBlockFwdOp':scale=scale.repeat_interleave(128,dim=1)[:,:q.shape[1]]
        return child.implementation.run(q.float(),scale).to(getattr(torch,p['out_dtype']))
    return Candidate(Implementation(run),'ops-math::Mul (symmetric INT8 dequantization)',
        'Exact INT8-to-FP32 widening; per-call per-tensor, per-row or 128-element block scale layout; native FP32 multiplication and manifest output cast.',
        child.source_sha256|source_hashes(Path(__file__)),cleanup=child.cleanup)
