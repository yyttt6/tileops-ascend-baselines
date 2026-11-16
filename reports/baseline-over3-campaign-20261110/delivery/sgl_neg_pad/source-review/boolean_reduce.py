"""Boolean reductions through complete original native selection/GEMV APIs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import UnsupportedCase, source_hashes
    from .reduction import argmax, row_layout
    from .selection_reduce import argmin
    x = case.inputs[0]
    if x.dtype != torch.bool:
        raise UnsupportedCase('This route requires boolean inputs; converting numeric inputs to truth values would replace missing original SGL predicate arithmetic.')
    name = case._call.signature.name
    pack, width, shape = row_layout(case)
    if name == 'CountNonzeroFwdOp':
        from .row_gemv import build as gemv
        metadata = SimpleNamespace(inputs=(torch.empty(x.shape, device=x.device, dtype=torch.float16),),
                                   params=case.params | {'dtype':'float32'})
        native = gemv(metadata, average=False)
        def run(value):
            return native.implementation.run(value.to(torch.float16)).to(torch.int64)
        description = 'Boolean 0/1 to FP16 is exact; original complete-row SGEMMV sums in FP32 and exact integer output is cast to int64. No external predicate or reduction.'
    else:
        native = argmin(case) if name == 'AllFwdOp' else argmax(case)
        def run(value):
            indices = native.implementation.run(value.to(torch.float32)).reshape(-1,1)
            return pack(value).gather(1,indices).reshape(shape)
        description = 'Losslessly widen original boolean inputs to FP32 0/1; complete native minimum/maximum selects an original boolean payload, with all original reduction axes retained.'
    return replace(native, implementation=Implementation(run),
                   adaptation=description, source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
