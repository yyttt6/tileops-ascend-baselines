"""Further classic adapters, separate from the sample currently under validation."""

from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, WORKSPACE, _add_layout,
    _owned_native, local_function, source_hashes,
)


def sgl_gqa(case):
    import torch
    from benchmarks.api import Implementation

    q, k, v, *extra = case.inputs
    p = case.params
    if any(t is not None for t in extra) or q.shape[1] != 1:
        raise UnsupportedCase("This adapter specializes the original one-query decode API.")
    if p.get("softcap") is not None or p.get("pos_encoding_mode", "none") != "none":
        raise UnsupportedCase("Decode API does not include softcap or RoPE.")
    if p.get("window_size_left", -1) != -1 or p.get("window_size_right", -1) != -1:
        raise UnsupportedCase("Decode API requires full visible KV history.")
    batch, _, heads, dim = q.shape
    seq, kv_heads, vdim = k.shape[1], k.shape[2], v.shape[-1]
    page_size = 128  # physical cache layout parameter of the original public API
    if seq % page_size:
        raise UnsupportedCase("This first adapter requires an integral number of cache pages.")
    pages = seq // page_size
    table = torch.arange(batch * pages, device=q.device, dtype=torch.int32).view(batch, pages)
    lengths = torch.full((batch,), seq, dtype=torch.int32, device=q.device)
    out = torch.empty((batch, heads, vdim), device=q.device, dtype=q.dtype)
    scale = p.get("sm_scale")
    scale = dim ** -0.5 if scale is None else scale
    host = "python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py"
    fn = local_function("sgl-kernel-npu", "python/sgl_kernel_npu",
                        "sgl_kernel_npu.attention.decode_attention", "decode_gqa", host)

    def run(q, k, v, *extra):
        fn(q.contiguous().view(batch, heads, dim),
           k.contiguous().view(batch * pages, page_size, kv_heads, dim),
           v.contiguous().view(batch * pages, page_size, kv_heads, vdim),
           out, lengths, scale, page_size, table)
        # A single bottom-right causal query sees the complete KV history.
        return out.view(batch, 1, heads, vdim)

    return Candidate(
        Implementation(run), "sgl_kernel_npu.attention.decode_attention.decode_gqa",
        "Dense KV -> paged view and static page metadata; one-query causal mask equals full history.",
        source_hashes(Path(__file__), LIBRARIES / "sgl-kernel-npu" / host),
    )


def sgl_deltanet(case):
    """Ungated specialization of SGL's existing fused GDN API."""
    import torch
    from benchmarks.api import Implementation

    q, k, v, beta, initial, offsets, offsets_cpu = case.inputs
    if offsets is not None or offsets_cpu is not None:
        raise UnsupportedCase("This pilot adapter handles dense batches only.")
    if case.params.get("use_qk_l2norm_in_kernel", False):
        raise UnsupportedCase("This SGL entry requires already-normalized Q/K.")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, beta)):
        raise UnsupportedCase("SGL's native fused entry requires BF16 Q/K/V/beta.")
    batch, seq, heads, dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if max(dim, value_dim) > 128 or max(heads, value_heads) > 64:
        raise UnsupportedCase("SGL's public entry limits dimensions to 128 and heads to 64.")
    lengths = torch.full((batch,), seq, dtype=torch.int32, device=q.device)
    zero_state = torch.zeros((batch, value_heads, value_dim, dim),
                             dtype=torch.float32, device=q.device)
    scale = case.params.get("scale")
    scale = dim ** -0.5 if scale is None else scale
    root = LIBRARIES / "sgl-kernel-npu"
    host = "python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk_gated_delta_rule_npu.py"
    fn = local_function("sgl-kernel-npu", "python/sgl_kernel_npu",
                        "sgl_kernel_npu.fla.chunk_gated_delta_rule_npu",
                        "chunk_gated_delta_rule_npu", host)

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        state = zero_state if initial is None else initial.transpose(-1, -2).contiguous()
        output, final = fn(
            q.contiguous().view(batch * seq, heads, dim),
            k.contiguous().view(batch * seq, heads, dim),
            v.contiguous().view(batch * seq, value_heads, value_dim),
            beta=beta.contiguous().view(batch * seq, value_heads),
            initial_state=state, actual_seq_lengths=lengths, scale=scale, g=None,
        )
        return (output.view(batch, seq, value_heads, value_dim),
                final.transpose(-1, -2).contiguous())

    sources = sorted((root / "csrc/chunk_gated_delta_rule").rglob("*.h"))
    sources += sorted((root / "csrc/chunk_gated_delta_rule").rglob("*.cpp"))
    return Candidate(
        Implementation(run), "sgl_kernel_npu.fla.chunk_gated_delta_rule_npu(g=None)",
        "Flatten batch/sequence; supply zero FP32 initial state and lengths; transpose final state.",
        source_hashes(
            Path(__file__), root / host, root / "csrc/pytorch_extensions.cpp",
            root / "python/sgl_kernel_npu/sgl_kernel_npu/__init__.py",
            root / "python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so",
            *sources,
        ),
        limitations=("Dense BF16 pilot only; every output, including FP32 final state, must pass.",),
    )


def fla_deltanet(case):
    """Use FLA's existing full GDN with zero decay and padded feature axes."""
    import importlib
    import os
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    q, k, v, beta, initial, offsets, offsets_cpu = case.inputs
    if (q.shape[1] == 1 and q.dtype == torch.bfloat16
            and not case.params.get("use_qk_l2norm_in_kernel", False)):
        from benchmarks.ascend_fla_recurrent import fla_recurrent_deltanet
        return fla_recurrent_deltanet(case)
    dim, value_dim = q.shape[-1], v.shape[-1]
    if max(dim, value_dim) > 128:
        raise UnsupportedCase("This adapter pads feature axes up to the original 128.")
    scale = case.params.get("scale")
    scale = dim ** -0.5 if scale is None else scale
    root = LIBRARIES / "flash-linear-attention-npu"
    source_package = root / "torch_custom/fla_npu/fla_npu"
    installed = Path(os.environ.get("ASCEND_CLASSIC_FLA_PACKAGE", str(WORKSPACE / ".venv/lib/python3.12/site-packages/fla_npu"))).resolve()
    # The original package requires its embedded OPP, so use the deployed copy
    # only after proving that every Python file is the inspected source version.
    python_sources = sorted(source_package.rglob("*.py"))
    for source in python_sources:
        deployed = installed / source.relative_to(source_package)
        if not deployed.is_file() or deployed.read_bytes() != source.read_bytes():
            raise RuntimeError(f"FLA deployment/source version mismatch: {deployed}")
    binary = installed / "opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
    symbols = ["aclnnChunkGatedDeltaRuleFwd", "aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize"]
    _owned_native(binary, symbols)  # A stock CANN dependency cannot stand in for this library.
    # This is an original public backend selector; it changes no kernel schedule.
    os.environ["FLA_NPU_STABLE_ABI"] = "ctypes"
    public = importlib.import_module("fla_npu.ops.ascendc")
    if Path(public.__file__).resolve() != (installed / "ops/ascendc/__init__.py").resolve():
        raise RuntimeError("FLA resolved outside the audited deployed package.")
    fn = public.npu_chunk_gated_delta_rule_fwd
    gate = torch.zeros(beta.shape, dtype=torch.float32, device=q.device)

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        # Zero padding preserves the recurrence. Explicit scale remains based on
        # the original feature dimension; padding/cropping costs are measured.
        initial_padded = None if initial is None else F.pad(
            initial, (0, 128 - value_dim, 0, 128 - dim))
        # Original API requires sequence-major chunk metadata with offsets.
        # This only describes the supplied ragged layout; native GDN handles it.
        metadata = offsets_cpu if offsets_cpu is not None else offsets
        cu = None if metadata is None else tuple(int(z) for z in metadata.detach().cpu().tolist())
        chunks = None if cu is None else tuple(
            item for seq, (begin, end) in enumerate(zip(cu, cu[1:]))
            for chunk in range((end - begin + 63) // 64) for item in (seq, chunk))
        qp, kp, vp = F.pad(q, (0, 128 - dim)), F.pad(k, (0, 128 - dim)), F.pad(v, (0, 128 - value_dim))
        native_gate, native_beta = gate, beta
        native_cu, native_chunks = cu, chunks
        if cu is not None:
            if q.shape[0] != 1 or cu[0] != 0 or cu[-1] != q.shape[1] or any(a >= b for a, b in zip(cu, cu[1:])):
                raise UnsupportedCase("Packed offsets must describe positive independent sequences.")
            longest = max(b - a for a, b in zip(cu, cu[1:]))
            def sequence_batch(tensor):
                return torch.stack([F.pad(tensor[0, a:b], (0, 0, 0, 0, 0, longest - (b - a)))
                                    for a, b in zip(cu, cu[1:])])
            qp, kp, vp = [sequence_batch(t) for t in (qp, kp, vp)]
            native_beta = torch.stack([F.pad(beta[0, a:b], (0, 0, 0, longest - (b - a)))
                                       for a, b in zip(cu, cu[1:])])
            native_gate = torch.zeros(native_beta.shape, device=q.device, dtype=torch.float32)
            # Independent sequences become independent dense batch entries.
            # Zero-beta/zero-QKV padding leaves the final recurrence state unchanged.
            native_cu, native_chunks = None, None
        layout = "BSND"
        if q.dtype == torch.float16:
            # Original A5 FP16 legacy path requires head-major Q/K/V. Its
            # gate, beta and output contracts remain sequence-major BSND.
            qp, kp, vp = [x.transpose(1, 2).contiguous() for x in (qp, kp, vp)]
            layout = "BNSD"
        outputs = fn(
            qp, kp, vp, native_gate, native_beta,
            initial_state=initial_padded, output_final_state=True,
            scale=scale, layout=layout, state_v_first=False,
            cu_seqlens=native_cu, chunk_indices=native_chunks,
            use_qk_l2norm_in_kernel=case.params.get("use_qk_l2norm_in_kernel", False),
        )
        if public.BACKENDS.get("npu_chunk_gated_delta_rule_fwd") != "ctypes":
            raise RuntimeError("FLA did not use the audited original ctypes backend.")
        output = outputs[0]
        if cu is not None:
            output = torch.cat([output[i, :b-a] for i, (a, b) in enumerate(zip(cu, cu[1:]))], dim=0).unsqueeze(0)
        return (output[..., :value_dim].contiguous(),
                outputs[1][..., :dim, :value_dim].contiguous())

    native = root / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd"
    sources = sorted(p for p in native.rglob("*") if p.suffix in (".h", ".hpp", ".cpp"))
    return Candidate(
        Implementation(run), "fla_npu.ops.ascendc.npu_chunk_gated_delta_rule_fwd(g=0)",
        "Pad K/V to128 and crop; preserve original scale, initial/final FP32 state and native L2Norm; pack ragged sequences as independently padded dense batches, and use head-major QKV for the original FP16 API.",
        source_hashes(Path(__file__), binary, *python_sources, *sources),
        limitations=("Requires the library's own deployed GDN entry; no stock-library fallback.",),
    )


def atvc_add(case):
    import ctypes as C
    import hashlib
    import os
    import shutil
    import subprocess
    import torch
    from benchmarks.api import Implementation

    x, _, shape, _, _, pack = _add_layout(case)
    suffixes = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}
    if x.dtype not in suffixes:
        raise UnsupportedCase("The existing AddCustom template is bound for FP16/BF16/FP32.")
    symbol = "classic_atvc_add_" + suffixes[x.dtype]
    root = LIBRARIES / "atvc"
    bridge = Path(__file__).parent / "native/ascend/atvc_add.cpp"
    headers = sorted((root / "include").rglob("*.h"))
    host_header = root / "examples/ops_pytorch/add/add_custom_impl.h"
    hashes = source_hashes(Path(__file__), bridge, host_header, *headers)
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("The installed Ascend compiler is unavailable.")
    digest = hashlib.sha256(repr(sorted(hashes.items())).encode()).hexdigest()
    cache = WORKSPACE / ".cache/ascend-classic" / ("atvc-add-" + digest[:20])
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "libclassic_atvc_add.so"
    if not binary.exists():
        sdk = Path(os.environ["ASCEND_HOME_PATH"])
        temporary = cache / "libclassic_atvc_add.building.so"
        command = [compiler, "-x", "asc", "--npu-arch=dav-3510",
                   "-shared", "-fPIC", str(bridge), "-o", str(temporary),
                   "-I", str(root / "include"), "-I", str(host_header.parent),
                   "-L", str(sdk / "lib64"), "-ltiling_api", "-lplatform",
                   "-lascendcl", "-lm", "-ldl"]
        result = subprocess.run(command, capture_output=True, text=True)
        (cache / "build.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"ATVC original-template compilation failed: {cache / 'build.log'}")
        temporary.replace(binary)
    lib = _owned_native(binary, [symbol])
    native_add = getattr(lib, symbol)
    native_add.argtypes = [
        C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64, C.c_void_p,
    ]
    native_add.restype = C.c_int
    out = torch.empty(shape, dtype=x.dtype, device=x.device)

    def run(x, y):
        a, b = pack(x), pack(y)
        rc = native_add(
            a.data_ptr(), b.data_ptr(), out.data_ptr(), out.numel(),
            torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f"ATVC host bridge returned {rc}")
        return out

    return Candidate(
        Implementation(run), "ATVC::AddCustom<" + str(x.dtype) + ">",
        "Instantiate original generic Add template; thin C ABI; original tiling and buffer defaults.",
        hashes | source_hashes(binary),
    )
