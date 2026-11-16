"""Repeat the original ops-tensor64-element Add over complete input chunks."""
import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _add_layout, _owned_native, source_hashes,
)


def opstensor_add(case):
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import torch
    from benchmarks.api import Implementation

    x, _, shape, rows, hidden, pack = _add_layout(case)
    if rows * hidden % 64:
        raise UnsupportedCase("Original64-element Add requires complete packed64-element chunks.")
    root = LIBRARIES / "ops-tensor"
    source = root / "include/tensor_api/examples/01_simd_cpp_api/07_tensor_api/experimental/reg_vector_compute/add/add.asc"
    bridge = Path(__file__).parent / "native/ascend/opstensor_add.cpp"
    headers = [p for p in (root / "include/tensor_api/include").rglob("*")
               if p.suffix in (".h", ".hpp")]
    hashes = source_hashes(Path(__file__), source, bridge, *headers)
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("Original ops-tensor Add requires bisheng.")
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    flags = ["-x", "asc", "--npu-arch=dav-3510", "-std=c++17", "-O2", "-shared", "-fPIC",
             "-DCANN_ASC_USE_EXPERIMENTAL"]
    includes = [root / "include/tensor_api/include", root / "include/tensor_api",
                sdk / "include", sdk / "x86_64-linux/asc/include", sdk / "asc/impl/c_api"]
    key = hashlib.sha256(json.dumps([hashes, flags, str(sdk.resolve())], sort_keys=True).encode()).hexdigest()[:20]
    cache = WORKSPACE / ".cache/ascend-classic" / ("opstensor-add-" + key)
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "libclassic.so"
    if not binary.exists():
        temporary = cache / "libclassic.building.so"
        command = [compiler, *flags, str(bridge), "-o", str(temporary)]
        for include in includes:
            command.extend(["-I", str(include)])
        command.extend(["-L", str(sdk / "lib64"), "-lascendcl", "-lascendc_runtime", "-lruntime"])
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        (cache / "build.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Original Add failed to build: {cache / 'build.log'}")
        temporary.replace(binary)
    lib = _owned_native(binary, ["classic_opstensor_add"])
    fn = lib.classic_opstensor_add
    fn.argtypes = [C.c_void_p] * 3 + [C.c_int64, C.c_void_p]
    fn.restype = C.c_int

    def run(a, b):
        lhs, rhs = pack(a).float(), pack(b).float()
        out = torch.empty_like(lhs)
        rc = fn(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), lhs.numel(),
                torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f"Original Add host status {rc}")
        return out.to(a.dtype).view(shape)

    return Candidate(
        Implementation(run), "ops-tensor::experimental/reg_vector_compute/add",
        "Broadcast/2D materialization; unchanged64-element kernel repeated by host loop; exact FP16/BF16 widening and output cast.",
        hashes | source_hashes(binary),
        limitations=("One original launch per64 values; measure complete wall time.",),
        timing_mode="host_roundtrip",
    )
