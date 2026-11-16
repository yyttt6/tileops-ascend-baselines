"""Original sign inversion plus complete native minimum selectors."""
from pathlib import Path
from .reduction import argmax,row_layout


def argmin(case):
    from dataclasses import replace
    from types import SimpleNamespace
    from benchmarks.api import Implementation
    from .elementwise import neg
    from benchmarks.ascend_baselines import source_hashes
    native=argmax(case)
    negative=neg(SimpleNamespace(inputs=case.inputs,params={}))
    def run(x):return native.implementation.run(negative.implementation.run(x))
    return replace(native,implementation=Implementation(run),api='sgl_kernel_npu::mul_add negative + native argmax',
                   adaptation='Native exact sign inversion then original first-index argmax on complete reduction rows; restore original output shape.',
                   source_sha256=native.source_sha256|negative.source_sha256|source_hashes(Path(__file__)))


def amin(case):
    from dataclasses import replace
    from benchmarks.api import Implementation
    native=argmin(case)
    pack,width,shape=row_layout(case)
    def run(x):
        indices=native.implementation.run(x).reshape(-1,1)
        return pack(x).gather(1,indices).reshape(shape)
    return replace(native,implementation=Implementation(run),api=native.api+' + gather original values',adaptation=native.adaptation+' Gather original input values at native argmin indices.')
