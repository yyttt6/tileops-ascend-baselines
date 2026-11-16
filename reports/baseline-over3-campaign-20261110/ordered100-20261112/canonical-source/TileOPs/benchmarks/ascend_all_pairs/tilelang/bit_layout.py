"""Bit-preserving dtype layouts and boolean-specialized original APIs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from .microblocks import _load_builder,clamp


def neg(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
    dtype=case.inputs[0].dtype
    if dtype not in (torch.float16,torch.bfloat16,torch.float32):raise UnsupportedCase('Sign-bit XOR layout covers the manifest FP16/BF16/FP32 formats only.')
    tl,builder,hashes=_load_builder('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','bitwise_kernel')
    hashes.update(source_hashes(Path(__file__)))
    kernel=tl.compile(builder(64),target='ascend')
    def run(x):
        flat=x.contiguous().flatten();count=flat.numel()
        if x.dtype!=torch.float32:flat=F.pad(flat,(0,count%2))
        bits=flat.view(torch.int32);n=bits.numel();bits=F.pad(bits,(0,-n%64))
        mask=torch.full((64,),-2147483648 if x.dtype==torch.float32 else -2147450880,device=x.device,dtype=torch.int32)
        out=torch.empty_like(bits);scratch=torch.empty(64,device=x.device,dtype=torch.int32)
        for start in range(0,bits.numel(),64):kernel(bits[start:start+64],mask,out[start:start+64],scratch)
        return out[:n].view(x.dtype)[:count].reshape(x.shape)
    return Candidate(Implementation(run),'tilelang::bitwise_kernel(64).Xor sign-bit mask',
                     'Lossless IEEE FP16/BF16/FP32 bit views; original int32 XOR flips only each sign bit. Independent slices, constant mask, padding/crop preserve zero/nonfinite representation. No Torch negation.',hashes)


def boolean_binary(case):
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
    # Inspect declared dtype before generating large unsupported nonboolean inputs.
    if case._call.tensors['input'][1]!='bool' or case._call.tensors['other'][1]!='bool':
        raise UnsupportedCase('Original clamp extrema implement boolean 0/1 And/Or only; nonboolean bitwise/logical route remains generic deployment-blocked.')
    op=case._call.signature.name
    select='MinimumFwdOp' if 'And' in op else 'MaximumFwdOp'
    proxy=SimpleNamespace(inputs=case.inputs,params=case.params,_call=SimpleNamespace(signature=SimpleNamespace(name=select)))
    native=clamp(proxy)
    return replace(native,api=native.api+' boolean '+select,
                   adaptation='Boolean operands widen exactly to FP32 0/1; original clamp min/max selects And/Or, then exact boolean output conversion. '+native.adaptation,
                   source_sha256=native.source_sha256|source_hashes(Path(__file__)))
