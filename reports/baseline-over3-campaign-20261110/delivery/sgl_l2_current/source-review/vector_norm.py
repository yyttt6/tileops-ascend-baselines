"""Norm values composed from complete unchanged SGL APIs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import UnsupportedCase, source_hashes
    from .scalar import selectors
    from .reduction import argmax, row_layout
    from .row_gemv import sum_reduce
    order = case.params['ord']
    if order == 2:
        from .l2_norm_value import build as l2
        return l2(case)
    if order not in (1, float('inf')):
        raise UnsupportedCase('Complete native compositions currently cover L1, L2 and infinity norms only.')
    abs_case = SimpleNamespace(inputs=case.inputs,params={},
        _call=SimpleNamespace(signature=SimpleNamespace(name='AbsFwdOp')))
    absolute = selectors(abs_case)
    pack, width, shape = row_layout(case)
    dtype = getattr(torch,case.params.get('dtype') or str(case.inputs[0].dtype).split('.')[-1])
    if order == 1:
        native = sum_reduce(case)
        def run(x):
            return native.implementation.run(absolute.implementation.run(x)).to(dtype)
    else:
        native = argmax(case)
        def run(x):
            values = absolute.implementation.run(x)
            indices = native.implementation.run(values).reshape(-1,1)
            return pack(values).gather(1,indices).reshape(shape).to(dtype)
    return replace(native,implementation=Implementation(run),
        api='SGL native sign/min/max absolute + complete native '+('sum' if order == 1 else 'argmax'),
        adaptation='Original SGL sign inversion and max implement absolute values; original complete-row sum or maximum implements L1/infinity norm. Input layout and final declared output conversion only; no Torch norm/reduction arithmetic.',
        source_sha256=absolute.source_sha256 | native.source_sha256 | source_hashes(Path(__file__)))
