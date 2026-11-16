"""Pointwise convolution using CATLASS's existing native GEMM bias option."""
import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes


def catlass_conv_with_bias(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_catlass_baselines import _native
    from benchmarks.ascend_catlass_bf16_conv import catlass_conv_all_types
    from benchmarks.ascend_pointwise_conv import _layout

    if not any(t is not None for t in case.inputs[2:]):
        return catlass_conv_all_types(case)
    x, weight, extra, pack, unpack = _layout(case)
    if x.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != x.dtype:
        raise UnsupportedCase('Original template requires matching FP16/BF16 operands.')
    if len(extra) != 1 or extra[0].dtype != x.dtype or tuple(extra[0].shape) != (weight.shape[0],):
        raise UnsupportedCase('Bias must contain one same-dtype value per output channel.')
    native, hashes = _native('gemm_bias', str(LIBRARIES / 'catlass/examples/43_ascend950_basic_matmul/basic_matmul_tla.cpp'),
                             'classic_catlass_gemm_bias')
    fn = native.classic_catlass_gemm_bias
    fn.argtypes = [C.c_void_p] * 4 + [C.c_uint32] * 3 + [C.c_int, C.c_void_p]
    fn.restype = C.c_int
    m = x.shape[0] * x.shape[2] * x.shape[3]
    n, k = weight.shape[:2]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    def run(x, weight, bias):
        a, b = pack(x, weight)
        b = b.T.contiguous()
        native_bias = bias.float().contiguous()
        rc = fn(a.data_ptr(), b.data_ptr(), native_bias.data_ptr(), out.data_ptr(), m, n, k,
                0 if x.dtype == torch.float16 else 1, torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f'CATLASS original bias GEMM host status {rc}')
        return unpack(out)

    return Candidate(Implementation(run), 'CATLASS example43 BasicMatmulTla native bias',
                     '1x1 NHWC matrix packing; exact FP32 bias widening into original optional bias input; original native template and scheduler choices; restore NCHW.',
                     hashes | source_hashes(Path(__file__), Path(__file__).with_name('ascend_pointwise_conv.py')))
