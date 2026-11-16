"""Bias as one constant input coordinate of the unchanged native convolution."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from benchmarks.ascend_baselines import UnsupportedCase, source_hashes


def _bias_coordinate(case, factory):
    import torch
    from benchmarks.api import Implementation

    x, weight, *extra = case.inputs
    if not any(t is not None for t in extra):
        return factory(case)
    if len(extra) != 1 or extra[0] is None:
        raise UnsupportedCase('One bias tensor is required for this coordinate encoding.')
    bias = extra[0]
    p = case.params
    pair = lambda z: (z, z) if isinstance(z, int) else tuple(z)
    kh, kw = weight.shape[-2:]
    dh, dw = pair(p['dilation'])
    if (p['groups'] != 1 or kh % 2 != 1 or kw % 2 != 1
            or pair(p['padding']) != (dh * (kh // 2), dw * (kw // 2))
            or weight.dtype != x.dtype or bias.dtype != x.dtype
            or tuple(bias.shape) != (weight.shape[0],)):
        raise UnsupportedCase('Constant-coordinate encoding requires centered odd kernels, same-dtype channel bias, and one group.')

    def pack(x, weight, bias):
        # The center tap always lands inside the input for this padding rule,
        # including border outputs and stride>1. Its constant input is exactly1.
        ones = torch.ones((x.shape[0], 1, *x.shape[2:]), dtype=x.dtype, device=x.device)
        coefficient = torch.zeros((weight.shape[0], 1, kh, kw), dtype=weight.dtype, device=weight.device)
        coefficient[:, 0, kh // 2, kw // 2].copy_(bias)
        return torch.cat((x, ones), dim=1), torch.cat((weight, coefficient), dim=1)

    packed = pack(x, weight, bias)
    native = factory(SimpleNamespace(inputs=packed, params=p))

    def run(x, weight, bias):
        return native.implementation.run(*pack(x, weight, bias))

    return replace(native, implementation=Implementation(run),
                   adaptation='Append a constant-one input channel and a zero filter channel containing bias only at its center tap. This is one original full convolution/GEMM on extended input coordinates; no external multiply/add or new kernel/schedule.',
                   source_sha256=native.source_sha256 | source_hashes(Path(__file__)),
                   limitations=(*native.limitations, 'All constant-channel packing counts in timing. Native extra-channel capacity and floating-point reduction order require device verification.'))


def catlass_bias_coordinate_cases(case):
    from benchmarks.ascend_catlass_bf16_conv import catlass_conv_all_types
    from benchmarks.ascend_catlass_bias_conv import catlass_conv_with_bias
    if tuple(case.inputs[1].shape[-2:]) == (1, 1):
        return catlass_conv_with_bias(case)
    return _bias_coordinate(case, catlass_conv_all_types)


def tensor_bias_coordinate_cases(case):
    from benchmarks.ascend_tensor_conv_cases import opstensor_conv_cases
    if tuple(case.inputs[1].shape[-2:]) == (1, 1):
        return opstensor_conv_cases(case)
    return _bias_coordinate(case, opstensor_conv_cases)


def tla_bias_coordinate_cases(case):
    from benchmarks.ascend_pointwise_conv import tla_pointwise_conv
    return _bias_coordinate(case, tla_pointwise_conv)


def tilelang_bias_coordinate_cases(case):
    from benchmarks.ascend_tilelang_conv_cases import tilelang_conv_cases
    return _bias_coordinate(case, tilelang_conv_cases)


def sgl_bias_coordinate_cases(case):
    from benchmarks.ascend_pointwise_conv import _no_bias_gemm
    from benchmarks.ascend_sgl_gemv_cases import sgl_gemv_cases
    return _bias_coordinate(case, lambda c: _no_bias_gemm(c, sgl_gemv_cases))
