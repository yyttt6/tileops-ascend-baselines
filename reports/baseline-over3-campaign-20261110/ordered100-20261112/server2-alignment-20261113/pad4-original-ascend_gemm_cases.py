"""Complete-tile padding for original GEMMs; native bias for 1x1 convolution."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes


def _layout(case, tiles):
    import torch.nn.functional as F

    a, b = case.inputs
    ta, tb = case.params['trans_a'], case.params['trans_b']
    m, k = a.T.shape if ta else a.shape
    kb, n = b.T.shape if tb else b.shape
    if k != kb or a.dtype != b.dtype:
        raise UnsupportedCase('Native dense GEMM requires equal reduction extents and operand dtypes.')
    bm, bn, bk = tiles
    def pack(a, b):
        lhs = (a.T if ta else a).contiguous()
        rhs = (b if tb else b.T).contiguous()
        return (F.pad(lhs, (0, -k % bk, 0, -m % bm)),
                F.pad(rhs, (0, -k % bk, 0, -n % bn)))
    return m, n, k, pack


def tilelang_gemm_cases(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_tilelang_baselines import _runtime, _runtime_hashes

    a, b = case.inputs
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise UnsupportedCase('This adapter selects original FP16/BF16 templates.')
    # The original window swizzle revisits tiles when its final M window
    # is larger than four rows. Pad whole zero rows to a full native window.
    logical_m = a.shape[1] if case.params['trans_a'] else a.shape[0]
    m_alignment = 1024 if logical_m > 1024 else 256
    m, n, k, pack = _layout(case, (m_alignment, 256, 256))
    tilelang = _runtime()
    host = 'examples/ascend/example_gemm.py'
    builder = local_function('tilelang', 'examples/ascend', 'example_gemm', 'gemm', host)
    dtype = str(a.dtype).split('.')[-1]
    output_dtype = 'float32' if a.dtype == torch.float16 else dtype
    kernel = tilelang.compile(builder(m + -m % m_alignment, k + -k % 256, n + -n % 256,
                                     dtype=dtype, out_dtype=output_dtype), target='ascend', out_idx=-1)
    def run(a, b):
        return kernel(*pack(a, b))[:m, :n].to(a.dtype).contiguous()
    return Candidate(Implementation(run), 'tilelang.examples.ascend.example_gemm.gemm',
                     'Original input/output dtype templates and default schedule; FP16 uses native FP32 output followed by manifest output cast; per-call transpose, complete-window M padding, complete-tile N/K padding and crop.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_tilelang_baselines.py'),
                                   LIBRARIES / 'tilelang' / host) | _runtime_hashes(tilelang))


def tla_gemm_cases(case):
    from benchmarks.api import Implementation
    from benchmarks.ascend_tla_baselines import tla_gemm

    m, n, _, pack = _layout(case, (128, 256, 64))
    native = tla_gemm(SimpleNamespace(inputs=pack(*case.inputs), params={'trans_a': False, 'trans_b': True}))
    def run(a, b):
        return native.implementation.run(*pack(a, b))[:m, :n].contiguous()
    return replace(native, implementation=Implementation(run),
                   adaptation='Per-call transpose and zero padding to original 128x256x64 tiles; crop output; unchanged native kernel and schedule.',
                   source_sha256=native.source_sha256 | source_hashes(Path(__file__)))


def deepgemm_pointwise_conv_cases(case):
    import sys
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import deepgemm_gemm
    from benchmarks.ascend_pointwise_conv import _layout as conv_layout

    x, weight, _, pack, unpack = conv_layout(case)
    a, b = pack(x, weight)
    native = deepgemm_gemm(SimpleNamespace(inputs=(a, b), params={'trans_a': False, 'trans_b': True}))
    module = sys.modules['deep_gemm']
    fn = module.bf16_gemm_nt
    if fn is not module._C.bf16_gemm_nt:
        raise RuntimeError('Native DeepGEMM bias call resolved to a different implementation.')
    out = torch.empty((a.shape[0], b.shape[0]), device=x.device, dtype=x.dtype)
    def run(x, weight, bias=None):
        a, b = pack(x, weight)
        fn(a, b, out, c=None if bias is None else bias.reshape(1, -1).contiguous())
        return unpack(out)
    return replace(native, implementation=Implementation(run),
                   adaptation='1x1 convolution: NHWC matrix view, original BF16 GEMM native optional [1,N] bias, restore NCHW; unchanged schedule.',
                   source_sha256=native.source_sha256 | source_hashes(Path(__file__), Path(__file__).with_name('ascend_pointwise_conv.py')))


def catlass_gemm_cases(case):
    import ctypes as C
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_catlass_baselines import _native

    a, b = case.inputs
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise UnsupportedCase('Original CATLASS template requires matching FP16/BF16 operands.')
    ta, tb = case.params['trans_a'], case.params['trans_b']
    m, k = a.T.shape if ta else a.shape
    kb, n = b.T.shape if tb else b.shape
    if k != kb:
        raise UnsupportedCase('Native GEMM reduction extents must agree.')
    transpose_result = m > n
    native_m, native_n = (n, m) if transpose_result else (m, n)
    kind, symbol = ('gemm_f16', 'classic_catlass_gemm_f16') if a.dtype == torch.float16 else ('gemm', 'classic_catlass_gemm_bf16')
    lib, hashes = _native(kind, '43_ascend950_basic_matmul/ascend950_basic_matmul_impl.cpp', symbol)
    fn = getattr(lib, symbol)
    fn.argtypes = [C.c_void_p] * 3 + [C.c_uint32] * 3 + [C.c_void_p]
    fn.restype = C.c_int
    out = torch.empty((native_m, native_n), device=a.device, dtype=a.dtype)
    def run(a, b):
        lhs, rhs = (a.T if ta else a), (b.T if tb else b)
        if transpose_result:
            lhs, rhs = rhs.T, lhs.T
        lhs, rhs = lhs.contiguous(), rhs.contiguous()
        rc = fn(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), native_m, native_n, k,
                torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f'Original CATLASS GEMM host status {rc}')
        return out.T.contiguous() if transpose_result else out
    return Candidate(Implementation(run), 'CATLASS example43 BasicMatmulTla',
                     'Original FP16/BF16 template; transpose operands/output when needed via (AB)^T=B^T A^T; retain existing M<=N scheduler and every kernel instruction.',
                     hashes | source_hashes(Path(__file__)))
