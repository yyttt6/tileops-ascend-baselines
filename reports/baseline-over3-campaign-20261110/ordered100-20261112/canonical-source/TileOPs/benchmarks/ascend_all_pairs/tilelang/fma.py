"""Constant-operand selection of the original SIMD fused multiply-add API."""
from pathlib import Path
from .microblocks import _load_builder


def binary(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase('Original SIMD FMA route accepts only losslessly widened floating operands.')
    subtraction=case._call.signature.name=='SubFwdOp'
    if subtraction and case.params.get('alpha',1)!=1:
        raise UnsupportedCase('This subtraction mapping is validated only for exact constant alpha=1.')
    tl,builder,hashes=_load_builder('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','test_fused_multiply_add','kernel')
    kernel=tl.compile(builder('vmula','float32',1e-5),target='ascend')
    hashes.update(source_hashes(Path(__file__)))
    def run(x,y):
        shape=tuple(torch.broadcast_shapes(x.shape,y.shape))
        a=x.expand(shape).contiguous().float().flatten()
        b=y.expand(shape).contiguous().float().flatten()
        count=a.numel()
        a=F.pad(a,(0,-count%128));b=F.pad(b,(0,-count%128))
        if subtraction:
            out=a.clone();a=b;b=torch.full_like(a,-1.)
        else:out=torch.zeros_like(a)
        for start in range(0,a.numel(),128):
            kernel(a[start:start+128],b[start:start+128],out[start:start+128])
        return out[:count].to(x.dtype).reshape(shape)
    return Candidate(Implementation(run),'tilelang::test_fused_multiply_add.kernel(vmula,float32)',
                     'Original SIMD FMA body and schedule unchanged. Multiplication selects a zero accumulator; subtraction selects current x accumulator and exact -1 multiplier for y. Broadcast packing, exact FP32 widening and output cast only; no Torch arithmetic. Inputs are copied before original in-place accumulator API.',hashes)
