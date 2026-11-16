"""Finite floating truth values computed through original native selectors."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import UnsupportedCase, source_hashes
    from .boolean_reduce import build as reduction
    from .boolean_elementwise import build as elementwise
    from .comparison import build as comparison
    name=case._call.signature.name
    factory=reduction if name in ('AnyFwdOp','AllFwdOp','CountNonzeroFwdOp') else elementwise
    if all(x.dtype == torch.bool for x in case.inputs):
        return factory(case)
    if any(x.dtype not in (torch.float16,torch.bfloat16,torch.float32) for x in case.inputs):
        raise UnsupportedCase('Finite native truth selection supports FP16/BF16/FP32 or boolean inputs; integer/complex predicate semantics are not replaced by a lossy cast.')
    metadata=SimpleNamespace(inputs=tuple(torch.empty(x.shape,device=x.device,dtype=torch.bool) for x in case.inputs),
                             params=case.params,_call=case._call)
    native=factory(metadata)
    truth=[]
    for x in case.inputs:
        zero=torch.zeros((),device=x.device,dtype=x.dtype)
        truth.append(comparison(SimpleNamespace(inputs=(x,zero),params={},
            _call=SimpleNamespace(signature=SimpleNamespace(name='NeFwdOp')))))
    def run(*values):
        flags=tuple(op.implementation.run(value,torch.zeros((),device=value.device,dtype=value.dtype))
                    for op,value in zip(truth,values,strict=True))
        return native.implementation.run(*flags)
    hashes=native.source_sha256 | source_hashes(Path(__file__))
    for op in truth:hashes.update(op.source_sha256)
    return replace(native,implementation=Implementation(run),source_sha256=hashes,
        api='SGL native finite nonzero comparison then '+native.api,
        adaptation='Original finite sparse/zero-row manifest values are compared with zero using both complete native first-tie selectors and original affine index sum; resulting boolean payloads enter the unchanged native boolean reduction/logic pipeline. No Torch predicate or target reduction arithmetic.',
        limitations=('Scoped to the original finite manifest value generators; integer/complex truth and arbitrary NaN predicate semantics are not claimed.',))
