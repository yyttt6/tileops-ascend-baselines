"""Existing SGL fused GDN with manifest dense or ragged metadata."""
from pathlib import Path
from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes


def sgl_delta_cases(case):
    """Ungated specialization of SGL's existing fused GDN API."""
    import torch
    from benchmarks.api import Implementation

    q, k, v, beta, initial, offsets, offsets_cpu = case.inputs
    if (q.dtype == torch.bfloat16 and not case.params.get("use_qk_l2norm_in_kernel", False)
            and q.shape[1] == 1):
        from benchmarks.ascend_sgl_kda import sgl_kda_deltanet
        return sgl_kda_deltanet(case)
    if case.params.get("use_qk_l2norm_in_kernel", False) or q.dtype == torch.float16:
        return sgl_delta_composite_cases(case)
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, beta)):
        raise UnsupportedCase("SGL's native fused entry requires BF16 Q/K/V/beta.")
    batch, seq, heads, dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if max(dim, value_dim) > 128 or max(heads, value_heads) > 64:
        raise UnsupportedCase("SGL's public entry limits dimensions to 128 and heads to 64.")
    scale = case.params.get("scale")
    scale = dim ** -0.5 if scale is None else scale
    root = LIBRARIES / "sgl-kernel-npu"
    host = "python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk_gated_delta_rule_npu.py"
    fn = local_function("sgl-kernel-npu", "python/sgl_kernel_npu",
                        "sgl_kernel_npu.fla.chunk_gated_delta_rule_npu",
                        "chunk_gated_delta_rule_npu", host)

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        metadata = offsets if offsets is not None else offsets_cpu
        lengths = (torch.full((batch,), seq, dtype=torch.int32, device=q.device)
                   if metadata is None else (metadata[1:] - metadata[:-1]).to(device=q.device, dtype=torch.int32))
        state = (torch.zeros((lengths.numel(), value_heads, value_dim, dim),
                             dtype=torch.float32, device=q.device)
                 if initial is None else initial.transpose(-1, -2).contiguous())
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
        "Flatten physical batch/sequence; convert supplied sequence offsets to native lengths, preserve initial FP32 states or allocate zero states; transpose native final state.",
        source_hashes(
            Path(__file__), root / host, root / "csrc/pytorch_extensions.cpp",
            root / "python/sgl_kernel_npu/sgl_kernel_npu/__init__.py",
            root / "python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so",
            *sources,
        ),
        limitations=("Original BF16 fused GDN API; every output, including FP32 final state, must pass.",),
    )


def sgl_delta_composite_cases(case):
    """Call SGL's existing complete Triton GDN entry with its native L2 flag."""
    import importlib
    import inspect
    import os
    import sys
    import torch
    from benchmarks.api import Implementation

    if os.environ.get('GDN_USE_MEGA_GDN', '0') != '0':
        raise UnsupportedCase('This receipt requires the original default SGL GDN pipeline.')
    root = LIBRARIES / 'sgl-kernel-npu'
    package = root / 'python/sgl_kernel_npu'
    source = package / 'sgl_kernel_npu/fla/chunk.py'
    sys.path.insert(0, str(package))
    module = importlib.import_module('sgl_kernel_npu.fla.chunk')
    fn = module.chunk_gated_delta_rule_npu
    if (Path(module.__file__).resolve() != source.resolve()
            or Path(inspect.unwrap(fn).__code__.co_filename).resolve() != source.resolve()):
        raise RuntimeError('SGL full GDN did not resolve to the original source and decorators.')
    files = sorted((package / 'sgl_kernel_npu/fla').glob('*.py'))
    for name, imported in tuple(sys.modules.items()):
        if name.startswith('sgl_kernel_npu.fla.') and getattr(imported, '__file__', None):
            if not Path(imported.__file__).resolve().is_relative_to(package.resolve()):
                raise RuntimeError(f'SGL dependency resolved outside original source: {name}')

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        cu = offsets if offsets is not None else (
            None if offsets_cpu is None else offsets_cpu.to(q.device))
        sequences = q.shape[0] if cu is None else cu.numel() - 1
        state = initial if initial is not None else torch.zeros(
            (sequences, v.shape[2], q.shape[3], v.shape[3]),
            device=q.device, dtype=torch.float32)
        original_shape = v.shape
        if cu is None:
            # Original WY kernel requires its existing packed-sequence path.
            # Flatten independent dense sequences and retain explicit boundaries.
            batch, length = q.shape[:2]
            cu = torch.arange(batch + 1, device=q.device, dtype=torch.int64) * length
            q, k, v = [t.contiguous().view(1, batch * length, *t.shape[2:])
                       for t in (q, k, v)]
            beta = beta.contiguous().view(1, batch * length, beta.shape[-1])
        gate = torch.zeros(beta.shape, dtype=torch.float32, device=q.device)
        out, final, _ = fn(
            q.contiguous(), k.contiguous(), v.contiguous(), gate, beta.contiguous(),
            scale=case.params.get('scale'), initial_state=state,
            output_final_state=True, cu_seqlens=cu, head_first=False,
            use_qk_l2norm_in_kernel=case.params.get('use_qk_l2norm_in_kernel', False))
        return out.view(original_shape), final

    return Candidate(
        Implementation(run), 'sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_npu(g=0)',
        'Original complete GDN pipeline and native L2Norm; flatten dense batches with explicit packed offsets, restore output layout, retain FP32 states and zero decay; no stage or schedule edits.',
        source_hashes(Path(__file__), *files, package / 'sgl_kernel_npu/__init__.py'),
        limitations=('Both complete outputs and preserved inputs must pass original tolerances.',),
    )

