"""Lossless original integer/complex logical manifest inputs and native truth."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes, UnsupportedCase
    from .finite_truth import build as finite
    from .boolean_reduce import build as reduce
    from .boolean_elementwise import build as boolean
    from .comparison import build as compare
    x, = case.inputs
    if x.dtype in (torch.bool, torch.float16, torch.bfloat16, torch.float32):
        return finite(case)
    if x.dtype not in (torch.int32, torch.int64, torch.complex64, torch.complex128):
        raise UnsupportedCase('No lossless original logical-input conversion for this dtype')
    flags = torch.empty(x.shape, device=x.device, dtype=torch.bool)
    meta = SimpleNamespace(inputs=(flags,), params=case.params, _call=case._call)
    native = reduce(meta)
    f32 = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    truth = compare(SimpleNamespace(inputs=(f32, f32), params={},
                    _call=SimpleNamespace(signature=SimpleNamespace(name='NeFwdOp'))))
    either = boolean(SimpleNamespace(inputs=(flags, flags), params={},
                     _call=SimpleNamespace(signature=SimpleNamespace(name='LogicalOrFwdOp')))) if x.is_complex() else None

    def nonzero(value):
        value = value.float()
        return truth.implementation.run(value, torch.zeros((), device=value.device, dtype=value.dtype))

    def run(value):
        # Original _make_logical_input draws integers in [-5,5], or constructs
        # both complex components in FP32 before optional complex128 widening.
        # Both conversions below are therefore exact for every original input.
        mask = (either.implementation.run(nonzero(value.real), nonzero(value.imag))
                if either is not None else nonzero(value))
        return native.implementation.run(mask)

    workload = Path(__file__).resolve().parents[3] / 'workloads/reduction.py'
    hashes = native.source_sha256 | truth.source_sha256 | source_hashes(Path(__file__), workload)
    if either is not None:
        hashes |= either.source_sha256
    return replace(native, implementation=Implementation(run), source_sha256=hashes,
        api='SGL native component nonzero comparison / boolean OR then ' + native.api,
        adaptation='Original logical generators draw bounded integers or FP32-valued real/imaginary components, including complex128 storage. Losslessly convert those components, compute each nonzero flag with unchanged original native selectors, combine complex components using native boolean OR, and execute the complete original native boolean reduction. All original axes and integer/boolean outputs retained.',
        limitations=('Complex128 support is scoped to the original FP32-valued component generator; arbitrary FP64 components, NaNs or other input contracts are not asserted. Actual original cases and unchanged exact comparator remain mandatory.',))
