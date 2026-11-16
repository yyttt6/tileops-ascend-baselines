"""Invoke original CATLASS templates through small host-only ABI wrappers."""

import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _owned_native, source_hashes,
)


def _native(kind, original, symbol):
    import hashlib
    import json
    import os
    import shutil
    import subprocess

    root = LIBRARIES / "catlass"
    optest = root / "tests/optest"
    bridge = Path(__file__).parent / f"native/ascend/catlass_{kind}.cpp"
    headers = []
    for include in (root / "include", optest / "include", optest / "kernels/common"):
        headers.extend(p for p in include.rglob("*") if p.suffix in (".h", ".hpp"))
    hashes = source_hashes(Path(__file__), bridge, optest / "kernels" / original, *headers)
    compiler = shutil.which("bisheng")
    if not compiler:
        raise RuntimeError("CATLASS bridge requires the installed bisheng compiler.")
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    flags = ["-x", "asc", "--npu-arch=dav-3510", "-std=c++17", "-O2",
             "-shared", "-fPIC", "-DCATLASS_ARCH=3510", "-DTILING_KEY_VAR",
             "-Wno-macro-redefined"]
    digest = hashlib.sha256(json.dumps(
        [hashes, compiler, str(sdk.resolve()), flags], sort_keys=True).encode()).hexdigest()
    cache = WORKSPACE / ".cache/ascend-classic" / f"catlass-{kind}-{digest[:20]}"
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "libclassic.so"
    if not binary.exists():
        temporary = cache / "libclassic.building.so"
        command = [compiler, *flags, str(bridge), "-o", str(temporary)]
        for include in (root / "include", optest / "include", optest / "kernels",
                        root / "examples/common"):
            command.extend(["-I", str(include)])
        command.extend(["-L", str(sdk / "lib64"), "-ltiling_api", "-lplatform", "-lascendcl", "-ldl"])
        result = subprocess.run(command, capture_output=True, text=True)
        (cache / "build.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Original CATLASS {kind} failed to build: {cache / 'build.log'}")
        temporary.replace(binary)
    return _owned_native(binary, [symbol]), hashes | source_hashes(binary)


def catlass_gemm(case):
    import torch
    from benchmarks.api import Implementation

    a, b = case.inputs
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise UnsupportedCase("This original template instantiation is BF16.")
    ta, tb = case.params["trans_a"], case.params["trans_b"]
    m, k = a.T.shape if ta else a.shape
    kb, n = b.T.shape if tb else b.shape
    if k != kb or m > n:
        raise UnsupportedCase("This pilot instantiates the original M<=N block-scheduler choice.")
    lib, hashes = _native("gemm", "43_ascend950_basic_matmul/ascend950_basic_matmul_impl.cpp",
                          "classic_catlass_gemm_bf16")
    fn = lib.classic_catlass_gemm_bf16
    fn.argtypes = [C.c_void_p] * 3 + [C.c_uint32] * 3 + [C.c_void_p]
    fn.restype = C.c_int
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    def run(a, b):
        a = (a.T if ta else a).contiguous()
        b = (b.T if tb else b).contiguous()
        rc = fn(a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k,
                torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f"CATLASS GEMM host status {rc}")
        return out

    return Candidate(Implementation(run), "CATLASS example43 BasicMatmulTla",
                     "Original BF16/row-major template; transpose/contiguous adapters; unchanged schedule.",
                     hashes)


def catlass_conv(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    x, weight, *extra = case.inputs
    p = case.params
    if any(v is not None for v in extra) or p.get("groups", 1) != 1:
        raise UnsupportedCase("The inspected original example has no bias or grouped convolution.")
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise UnsupportedCase("The original example-56 compute is instantiated for FP16.")
    batch, channels, hi, wi = x.shape
    out_channels, _, kh, kw = weight.shape
    stride = p.get("stride", (1, 1))
    stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pad = p.get("padding", (0, 0))
    pad = (pad, pad) if isinstance(pad, int) else tuple(pad)
    dilation = p.get("dilation", (1, 1))
    dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    ho = (hi + 2 * pad[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    wo = (wi + 2 * pad[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    ci1, co1 = (channels + 15) // 16, (out_channels + 15) // 16
    lib, hashes = _native("conv", "56_ascend950_basic_conv2d_tla/basic_conv2d_tla.cpp",
                          "classic_catlass_conv_f16")
    fn = lib.classic_catlass_conv_f16
    fn.argtypes = [C.c_void_p] * 3 + [C.POINTER(C.c_uint32), C.c_void_p]
    fn.restype = C.c_int
    sizes = (C.c_uint32 * 15)(
        batch, hi, wi, ci1 * 16, out_channels, kh, kw, *stride,
        pad[1], pad[1], pad[0], pad[0], *dilation)
    out = torch.empty((batch, co1, ho, wo, 16), device=x.device, dtype=x.dtype)

    def run(x, weight, *extra):
        x = F.pad(x, (0, 0, 0, 0, 0, ci1 * 16 - channels))
        weight = F.pad(weight, (0, 0, 0, 0, 0, ci1 * 16 - channels))
        x = x.reshape(batch, ci1, 16, hi, wi).permute(0, 1, 3, 4, 2).contiguous()
        weight = weight.reshape(out_channels, ci1, 16, kh, kw).permute(1, 3, 4, 0, 2).contiguous()
        rc = fn(x.data_ptr(), weight.data_ptr(), out.data_ptr(), sizes,
                torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f"CATLASS convolution host status {rc}")
        return out.permute(0, 1, 4, 2, 3).contiguous().view(batch, co1 * 16, ho, wo)[:, :out_channels]

    return Candidate(
        Implementation(run), "CATLASS example56 Ascend950BasicConv2dTLA",
        "NCHW/OIHW -> NC1HWC0/CI1KHKWCOCI0 and back; all copies inside run; original compute and schedule.",
        hashes | source_hashes(LIBRARIES / "catlass/tests/optest/tests/conv_common.py"),
    )
