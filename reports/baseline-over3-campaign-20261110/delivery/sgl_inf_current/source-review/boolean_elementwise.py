"""Exact boolean-domain forms of unchanged original SGL arithmetic APIs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import UnsupportedCase, source_hashes
    from .additional import mul, maximum
    from .elementwise import sub
    if any(x.dtype != torch.bool for x in case.inputs):
        raise UnsupportedCase('This original arithmetic composition accepts only boolean operands; numeric truth predicates and integer bit operations have no equivalent registered native stage.')
    name = case._call.signature.name
    if name in ('BitwiseAndFwdOp','LogicalAndFwdOp'):
        native = mul(case)
        def run(x,y):
            return native.implementation.run(x.float(),y.float()).to(torch.bool)
    elif name in ('BitwiseOrFwdOp','LogicalOrFwdOp'):
        native = maximum(case)
        def run(x,y):
            return native.implementation.run(x.float(),y.float()).to(torch.bool)
    elif name == 'BitwiseXorFwdOp':
        native = sub(case)
        def run(x,y):
            return native.implementation.run(x.float(),y.float()).to(torch.bool)
    else:
        metadata = SimpleNamespace(inputs=(case.inputs[0],case.inputs[0]),params={})
        native = sub(metadata)
        def run(x):
            values = x.float()
            return native.implementation.run(torch.ones_like(values),values).to(torch.bool)
    return replace(native,implementation=Implementation(run),
        adaptation='Lossless boolean 0/1 widening; complete original SGL product (AND), maximum (OR), difference (XOR), or constant-one difference (NOT). Every intermediate is exactly representable and the final boolean storage cast restores the declared output. No Torch logical/bitwise operator replaces target arithmetic.',
        source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
