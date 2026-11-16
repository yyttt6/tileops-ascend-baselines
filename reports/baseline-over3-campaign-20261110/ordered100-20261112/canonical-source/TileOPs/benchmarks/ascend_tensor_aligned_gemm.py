"""Shape-only padding around the original ops-tensor GEMM example."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from benchmarks.ascend_baselines import source_hashes


def opstensor_aligned_gemm(case):
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_tensor_gemm_cases import opstensor_gemm_cases

    a, b = case.inputs
    ta, tb = case.params['trans_a'], case.params['trans_b']
    rows, reduction = a.T.shape if ta else a.shape
    _, columns = b.T.shape if tb else b.shape
    # The original GetTilingConfig/ComputeTiling use M=N=256, K=128
    # for both sixteen-bit dtypes. Fill whole tiles without editing tiling,
    # tail scheduling, buffer counts, or the native kernel.
    pad_rows, pad_columns, pad_reduction = -rows % 256, -columns % 256, -reduction % 128

    def pack(a, b):
        a = (a.T if ta else a).contiguous()
        b = (b.T if tb else b).contiguous()
        return (F.pad(a, (0, pad_reduction, 0, pad_rows)),
                F.pad(b, (0, pad_columns, 0, pad_reduction)))

    native = opstensor_gemm_cases(SimpleNamespace(
        inputs=pack(a, b), params={'trans_a': False, 'trans_b': False}))

    def run(a, b, bias=None):
        packed_bias = None if bias is None else F.pad(bias, (0, pad_columns))
        value = native.implementation.run(*pack(a, b), bias=packed_bias)
        return value[:rows, :columns].contiguous()

    return replace(native, implementation=Implementation(run),
                   adaptation='Zero-pad original GEMM operands to complete 256x256x128 tiles; preserve full reduction and native bias; crop output. Native schedule unchanged.',
                   source_sha256=native.source_sha256 | source_hashes(Path(__file__)))


def opstensor_aligned_pointwise_conv(case):
    from benchmarks.api import Implementation
    from benchmarks.ascend_pointwise_conv import _layout

    x, weight, _, pack, unpack = _layout(case)
    native = opstensor_aligned_gemm(SimpleNamespace(
        inputs=pack(x, weight), params={'trans_a': False, 'trans_b': True}))

    def run(x, weight, bias=None):
        return unpack(native.implementation.run(*pack(x, weight), bias=bias))

    return replace(native, implementation=Implementation(run),
                   adaptation='1x1 convolution NHWC packing, whole-tile zero padding, original native GEMM/bias, crop and restore NCHW.',
                   source_sha256=native.source_sha256 | source_hashes(
                       Path(__file__).with_name('ascend_pointwise_conv.py')))


def opstensor_aligned_conv_cases(case):
    if tuple(case.inputs[1].shape[-2:]) == (1, 1):
        return opstensor_aligned_pointwise_conv(case)
    from benchmarks.ascend_bias_coordinate_conv import tensor_bias_coordinate_cases
    return tensor_bias_coordinate_cases(case)
