"""Bindings to ops-tensor's original executable examples and templates."""

import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _owned_native, source_hashes,
)


def opstensor_gemm(case):
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import torch
    from benchmarks.api import Implementation

    a, b = case.inputs
    if a.dtype != torch.bfloat16 or b.dtype != a.dtype:
        raise UnsupportedCase("This first original-template instantiation is BF16.")
    ta, tb = case.params["trans_a"], case.params["trans_b"]
    m, k = a.T.shape if ta else a.shape
    kb, n = b.T.shape if tb else b.shape
    if kb != k:
        raise UnsupportedCase("GEMM dimensions disagree.")
    root = LIBRARIES / "ops-tensor"
    source = root / "examples/mat_mul/mat_mul_basic/mat_mul_basic.cpp"
    bridge = Path(__file__).parent / "native/ascend/opstensor_gemm.cpp"
    headers = [p for p in (root / "include").rglob("*") if p.suffix in (".h", ".hpp")]
    hashes = source_hashes(Path(__file__), source, bridge,
                           root / "examples/common/data_utils.h", *headers)
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("Original ops-tensor templates require bisheng.")
    # Include/link directories follow this library's examples/CMakeLists.txt.
    includes = [
        root / "include", root / "include/blaze", root / "include/tensor_api/include",
        root / "include/tensor_api", root / "examples", root / "examples/common",
        sdk / "include", sdk / "include/op_common", sdk / "pkg_inc/op_common",
        sdk / "pkg_inc/base", sdk / "pkg_inc", sdk / "x86_64-linux/asc",
        sdk / "x86_64-linux/asc/include", sdk / "x86_64-linux/asc/include/tiling",
        sdk / "asc/impl/c_api", sdk / "asc/impl/utils",
        sdk / "compiler/ascendc/include/highlevel_api",
    ]
    flags = ["-x", "asc", "--npu-arch=dav-3510", "-std=c++17", "-O2", "-shared", "-fPIC"]
    libraries = ["tiling_api", "platform", "ascendcl", "ascendc_runtime", "runtime", "mmpa", "c_sec", "dl"]
    key = hashlib.sha256(json.dumps(
        [hashes, compiler, str(sdk.resolve()), flags, list(map(str, includes)), libraries],
        sort_keys=True,
    ).encode()).hexdigest()
    cache = WORKSPACE / ".cache/ascend-classic" / f"opstensor-gemm-{key[:20]}"
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "libclassic.so"
    if not binary.exists():
        temporary = cache / "libclassic.building.so"
        command = [compiler, *flags, str(bridge), "-o", str(temporary)]
        for include in includes:
            command.extend(["-I", str(include)])
        command.extend(["-iquote" + str(root / "include/tensor_api/include"),
                        "-iquote" + str(root / "include/tensor_api"),
                        "-L", str(sdk / "lib64"), "-L", str(sdk / "x86_64-linux/lib64")])
        command.extend("-l" + name for name in libraries)
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        (cache / "build.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Original ops-tensor example failed to build: {cache / 'build.log'}")
        temporary.replace(binary)
    lib = _owned_native(binary, ["classic_opstensor_gemm_bf16"])
    fn = lib.classic_opstensor_gemm_bf16
    fn.argtypes = [C.c_void_p] * 3 + [C.c_int64] * 3 + [C.c_void_p]
    fn.restype = C.c_int
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    def run(a, b):
        a, b = (a.T if ta else a).contiguous(), (b.T if tb else b).contiguous()
        rc = fn(a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k,
                torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f"ops-tensor GEMM host status {rc}")
        return out

    return Candidate(
        Implementation(run), "ops-tensor::mat_mul_basic (Blaze GemmUniversal)",
        "Original BF16 example template, original host tiling/schedule; direct pointers and per-call layout adapters.",
        hashes | source_hashes(binary),
    )
