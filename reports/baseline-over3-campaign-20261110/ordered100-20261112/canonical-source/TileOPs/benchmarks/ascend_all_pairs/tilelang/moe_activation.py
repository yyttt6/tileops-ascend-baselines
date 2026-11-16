"""Original per-expert GEMM followed by original native SiLU API composition."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path


def moe_grouped_dispatch(case):
    from .adapters import moe_grouped
    if case.params.get('activation') is None:return moe_grouped(case)
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
    from .composed import composed
    if case.params['activation']!='silu_and_mul':raise UnsupportedCase('Only current SiLU-and-multiply manifest epilogue has a native composition.')
    p=dict(case.params);p.update(activation=None,out_dtype='float32')
    gemm=moe_grouped(SimpleNamespace(inputs=case.inputs,params=p,_call=case._call))
    activation=composed(SimpleNamespace(params={},_call=SimpleNamespace(signature=SimpleNamespace(name='SiluAndMulFwdOp'))))
    dtype=getattr(torch,case.params['out_dtype']) if case.params.get('out_dtype') else case.inputs[0].dtype
    hashes=gemm.source_sha256|activation.source_sha256|source_hashes(Path(__file__))
    def run(*inputs):
        accumulator=gemm.implementation.run(*inputs)
        result=activation.implementation.run(accumulator).to(dtype)
        # Original GEMMs may be compiled lazily for current runtime metadata.
        hashes.update(gemm.source_sha256)
        return result
    return replace(gemm,implementation=Implementation(run),api=gemm.api+' + original native SiLU composition',
                   adaptation=gemm.adaptation+' Keep original FP32 GEMM output through native SiLU/gate multiplication and cast only at the end, matching the original oracle accumulation contract. Metadata and all intermediate operands are processed inside run.',source_sha256=hashes)
