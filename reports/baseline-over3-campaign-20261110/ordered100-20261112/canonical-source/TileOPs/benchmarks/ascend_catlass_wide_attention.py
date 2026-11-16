"""Original CATLASS example70: full QK reduction and independent V-column groups."""
import ctypes as C
from pathlib import Path
from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _owned_native, source_hashes


def catlass_wide_gqa_cases(case):
    if case.inputs[0].shape[-1] <= 128:
        from benchmarks.ascend_catlass_attention_cases import catlass_gqa_cases
        return catlass_gqa_cases(case)
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
    if (sq != 1 or dim != 256 or sk % 128 or k.shape != v.shape or hq % hk
            or q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype
            or any(t is not None for t in extra) or p.get('softcap') is not None
            or p.get('pos_encoding_mode', 'none') != 'none'
            or p.get('window_size_left', -1) != -1 or p.get('window_size_right', -1) != -1):
        raise UnsupportedCase('Focused example70 alternative requires complete-context256-wide16bit decode.')
    root = LIBRARIES / "catlass"
    example = root / "examples/70_ascend950_flash_attention_chunk_prefill"
    bridge = Path(__file__).parent / "native/ascend/catlass_attention70.cpp"
    files = [Path(__file__), bridge, example / "fai.cpp"]
    for directory in (root / "include", root / "examples/common", example):
        files.extend(f for f in directory.rglob("*") if f.suffix in (".h", ".hpp", ".cpp"))
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
    cache = WORKSPACE / ".cache/ascend-classic" / f"catlass-attention70-{key[:20]}"
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
            raise RuntimeError(f"CATLASS example70 build failed: {cache / 'build.log'}")
        temporary.replace(binary)
    lib = _owned_native(binary, ['classic_catlass70_tiling_size', 'classic_catlass70_prepare', 'classic_catlass70_run'])
    lib.classic_catlass70_tiling_size.restype = C.c_uint64
    lib.classic_catlass70_prepare.argtypes = [C.c_void_p, C.POINTER(C.c_int64), C.c_float, C.c_int]
    lib.classic_catlass70_prepare.restype = C.c_uint64
    lib.classic_catlass70_run.argtypes = [C.c_void_p] * 12 + [C.c_int]
    lib.classic_catlass70_run.restype = C.c_int
    host = C.create_string_buffer(lib.classic_catlass70_tiling_size())
    sizes = (C.c_int64 * 6)(batch, sq, sk, hq, hk, dim)
    dtype = 0 if q.dtype == torch.float16 else 1
    scale = dim ** -0.5 if p.get('sm_scale') is None else float(p['sm_scale'])
    workspace_size = lib.classic_catlass70_prepare(host, sizes, scale, dtype)
    if not workspace_size:
        raise RuntimeError('Original example70 host metadata rejected the requested dimensions.')
    tiling = torch.tensor(list(host.raw), dtype=torch.uint8, device=q.device)
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=q.device)
    pages = sk // 128
    table = torch.arange(batch * pages, dtype=torch.int32, device=q.device).reshape(batch, pages)
    q_lengths = torch.arange(1, batch + 1, dtype=torch.int64, device=q.device) * sq
    kv_lengths = torch.full((batch,), sk, dtype=torch.int64, device=q.device)
    positions = torch.arange(2048, device=q.device)
    mask = (positions[None, :] > positions[:, None]).to(torch.uint8).contiguous()
    count = batch * sq * hq * 128
    storage = [torch.empty(2 * count, dtype=q.dtype, device=q.device) for _ in range(2)]
    outputs = [t[:count].reshape(batch, sq, hq, 128) for t in storage]
    lse = torch.empty(2 * batch * sq * hq, dtype=torch.float32, device=q.device)

    def run(q, k, v, *extra):
        query = q.contiguous().reshape(batch * sq, hq, dim)
        keys = k.reshape(batch, pages, 128, hk, dim).permute(0, 1, 3, 2, 4).contiguous()
        for i, begin in enumerate(range(0, dim, 128)):
            values = v[..., begin:begin + 128].contiguous().reshape(batch, pages, 128, hk, 128)
            values = values.permute(0, 1, 3, 2, 4).contiguous()
            rc = lib.classic_catlass70_run(
                query.data_ptr(), keys.data_ptr(), values.data_ptr(), mask.data_ptr(),
                table.data_ptr(), storage[i].data_ptr(), lse.data_ptr(), q_lengths.data_ptr(),
                kv_lengths.data_ptr(), workspace.data_ptr(), tiling.data_ptr(),
                torch.npu.current_stream().npu_stream, dtype)
            if rc:
                raise RuntimeError(f'Original example70 launch status{rc}')
        return torch.cat(outputs, dim=-1)

    return Candidate(
        Implementation(run), 'CATLASS original example70 FAInfer (independent128-wide V groups)',
        'Original full256-wide QK reduction and native softmax recomputed for each independent128-wide V group; concatenate outputs. QK L1 tile extents only: KLeft192->256, KRight192->256, N256->128. All buffer counts, prelaunch, barriers and native algorithms remain unchanged.',
        hashes | source_hashes(binary),
        limitations=('Unverified original-entry alternative to example49 hard output-width limit. Original host tiling structure supports at most24active cores; all packing and both native attention calls count in timing.',),
    )
