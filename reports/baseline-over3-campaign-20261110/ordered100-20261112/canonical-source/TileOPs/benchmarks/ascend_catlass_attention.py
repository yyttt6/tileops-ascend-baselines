"""CATLASS's original Ascend950 attention example through a host ABI."""
import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _owned_native, source_hashes,
)


def catlass_gqa(case):
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import torch
    from benchmarks.api import Implementation

    q, k, v, *extra = case.inputs
    batch, sq, hq, dim = q.shape
    _, sk, hk, _ = k.shape
    p = case.params
    if (any(t is not None for t in extra)
            or q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype
            or sq != 1 or dim > 128 or hq % hk or k.shape != v.shape
            or p.get("softcap") is not None or p.get("pos_encoding_mode", "none") != "none"
            or p.get("window_size_left", -1) != -1 or p.get("window_size_right", -1) != -1):
        raise UnsupportedCase("This sample instantiates the original FP16 full-context decode path.")
    root = LIBRARIES / "catlass"
    example = root / "examples/49_ascend950_flash_attention_infer"
    bridge = Path(__file__).parent / "native/ascend/catlass_attention.cpp"
    files = [Path(__file__), bridge, example / "fai.cpp"]
    for directory in (root / "include", root / "examples/common", example):
        files.extend(f for f in directory.rglob("*") if f.suffix in (".h", ".hpp"))
    hashes = source_hashes(*files)
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("CATLASS requires bisheng.")
    flags = ["-x", "asc", "--npu-arch=dav-3510", "-std=c++17", "-O2",
             "-shared", "-fPIC", "-DCATLASS_ARCH=3510", "-DTILING_KEY_VAR",
             "-Wno-macro-redefined"]
    key = hashlib.sha256(json.dumps(
        [hashes, compiler, str(sdk.resolve()), flags], sort_keys=True).encode()).hexdigest()
    cache = WORKSPACE / ".cache/ascend-classic" / f"catlass-attention-{key[:20]}"
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "libclassic.so"
    if not binary.exists():
        temporary = cache / "libclassic.building.so"
        command = [compiler, *flags, str(bridge), "-o", str(temporary)]
        for include in (root / "include", root / "examples/common", example,
                        sdk / "include", sdk / "include/aclnn"):
            command.extend(["-I", str(include)])
        command.extend(["-L", str(sdk / "lib64"), "-ltiling_api", "-lplatform", "-lascendcl", "-ldl"])
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        (cache / "build.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"CATLASS example49 build failed: {cache / 'build.log'}")
        temporary.replace(binary)
    lib = _owned_native(binary, ["classic_catlass_fa_tiling_size",
                                 "classic_catlass_fa_prepare", "classic_catlass_fa_f16"])
    lib.classic_catlass_fa_tiling_size.restype = C.c_uint64
    lib.classic_catlass_fa_prepare.argtypes = [C.c_void_p, C.POINTER(C.c_int64), C.c_float]
    lib.classic_catlass_fa_prepare.restype = C.c_int
    lib.classic_catlass_fa_f16.argtypes = [C.c_void_p] * 8
    lib.classic_catlass_fa_f16.restype = C.c_int
    size = lib.classic_catlass_fa_tiling_size()
    host = C.create_string_buffer(size)
    sizes = (C.c_int64 * 6)(batch, sq, sk, hq, hk, dim)
    scale = p.get("sm_scale") if p.get("sm_scale") is not None else dim ** -0.5
    if lib.classic_catlass_fa_prepare(host, sizes, scale):
        raise RuntimeError("CATLASS original attention tiling rejected the shape.")
    tiling = torch.tensor(list(host.raw), dtype=torch.uint8, device=q.device)
    q_lengths = torch.full((batch,), sq, dtype=torch.int64, device=q.device)
    kv_lengths = torch.full((batch,), sk, dtype=torch.int64, device=q.device)
    # The original example allocates 2x output space. Preserve that allocation.
    storage = torch.empty(2 * q.numel(), dtype=q.dtype, device=q.device)
    out = storage[:q.numel()].view(q.shape)

    def run(q, k, v, *extra):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        rc = lib.classic_catlass_fa_f16(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), storage.data_ptr(),
            q_lengths.data_ptr(), kv_lengths.data_ptr(), tiling.data_ptr(),
            torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f"CATLASS attention host status {rc}")
        return out

    return Candidate(
        Implementation(run), "CATLASS example49 FAInferTla<half,false,false>",
        "BSND/TND contiguous input views; decode bottom-right causal mask is all-visible; original tiling and kernel.",
        hashes | source_hashes(binary),
    )
