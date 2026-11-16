"""Complete original SGL Softplus, Tanh and product APIs."""
from pathlib import Path


def build(case):
    from types import SimpleNamespace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .softplus import build as softplus
    from .extra_arithmetic import tanh
    from .additional import mul
    x, = case.inputs
    soft = softplus(SimpleNamespace(inputs=(x.float(),), params={'beta':1.0, 'threshold':20.0}))
    tangent, product = tanh(case), mul(case)
    def run(x):
        value = soft.implementation.run(x.float())
        value = tangent.implementation.run(value)
        return product.implementation.run(x.float(), value).to(x.dtype)
    return Candidate(Implementation(run), soft.api+' + '+tangent.api+' + '+product.api,
        'Lossless FP32 widening, complete original Softplus and native Tanh composition, original product, one final storage cast. No external activation arithmetic.',
        soft.source_sha256 | tangent.source_sha256 | product.source_sha256 | source_hashes(Path(__file__)))
