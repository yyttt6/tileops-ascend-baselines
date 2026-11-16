"""Original SGL full-FP8 sparse attention with dense causal page metadata."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, _owned_native, source_hashes


def sgl_fp8_gqa_cases(case):
    import importlib
    import sys
    import torch
    from benchmarks.api import Implementation

    if case.inputs[0].dtype != torch.float8_e4m3fn:
        from dataclasses import replace
        from benchmarks.ascend_rope_attention_cases import sgl_rope_gqa_cases
        native = sgl_rope_gqa_cases(case)
        return replace(native, source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
    if len(case.inputs) != 8:
        raise UnsupportedCase('FP8 attention requires Q/K/V, scales and optional rotation tables.')
    q, k, v, qs, ks, vs, cos, sin = case.inputs
    p = case.params
    batch, queries, heads, dim = q.shape
    keys, kvheads, vdim = k.shape[1:]
    page_size = 128
    pages = (keys + page_size - 1) // page_size
    if (batch != 1 or dim != 128 or vdim != dim or k.shape != v.shape
            or k.dtype != q.dtype or v.dtype != q.dtype or heads % kvheads
            or keys % page_size or pages > 16 or queries > keys or not p.get('is_causal')
            or p.get('softcap') is not None or p.get('pos_encoding_mode', 'none') != 'none'
            or p.get('window_size_left', -1) != -1 or p.get('window_size_right', -1) != -1
            or p.get('out_dtype') != 'float16' or cos is not None or sin is not None
            or any(s is None or s.dtype != torch.float32 or tuple(s.shape) != (batch, kvheads)
                   for s in (qs, ks, vs))):
        raise UnsupportedCase('Focused original full-FP8 causal sparse entry: one batch, D128, <=16 complete128-token pages, per-KV-head scales.')
    repeat_kv_heads = heads // kvheads
    # Express GQA as MHA by repeating only independent K/V heads. The native
    # entry now sees one query head per KV head; its kernel remains unchanged.
    kvheads = heads
    root = LIBRARIES / 'sgl-kernel-npu'
    package = root / 'python/sgl_kernel_npu/sgl_kernel_npu'
    sys.path.insert(0, str(package.parent))
    loaded = importlib.import_module('sgl_kernel_npu')
    if Path(loaded.__file__).resolve() != (package / '__init__.py').resolve():
        raise RuntimeError('SGL package differs from the inspected original source.')
    binary = package / 'lib/libsgl_kernel_npu.so'
    _owned_native(binary, ['_ZN6sglang10npu_kernel22sparse_attention_scoreERKN2at6TensorES4_S4_S4_S4_RKSt8optionalIS2_ES8_S8_S8_S8_S8_ldlll'])
    from torch.utils.cpp_extension import load
    bridge_source = Path(__file__).parent / 'native/ascend/sgl_sparse_attention_binding.cpp'
    build_dir = LIBRARIES.parent / '.cache/ascend-classic/sgl_sparse_binding_113'
    build_dir.mkdir(parents=True, exist_ok=True)
    bridge_binary = load(name='classic_sgl_fp8_binding', sources=[str(bridge_source)],
                        extra_include_paths=[str(root / 'csrc/sparse_attention_score/op_host'),
                                             str(root / 'csrc/utils')],
                        extra_ldflags=[str(binary), '-Wl,-rpath,' + str(binary.parent)],
                        extra_cflags=['-O2'], build_directory=str(build_dir),
                        is_python_module=False, verbose=False)
    table = torch.arange(pages, device=q.device, dtype=torch.int32).reshape(1, pages)
    # Native scale addressing uses logical KV positions and derives its Q-scale
    # stride from select_idx.shape[1]. Reserve the whole KV position extent;
    # only the first actualQ=queries rows are visited by its task loop.
    selections = torch.arange(pages, device=q.device, dtype=torch.int32).reshape(1, 1, pages)
    selections = selections.expand(kvheads, keys, pages).contiguous()
    valid = torch.zeros((kvheads, keys), device=q.device, dtype=torch.int32)
    visible = torch.arange(queries, device=q.device, dtype=torch.int32) + keys - queries + 1
    valid[:, :queries] = ((visible + page_size - 1) // page_size)[None, :]
    q_lengths = torch.tensor([queries], device=q.device, dtype=torch.int32)
    kv_lengths = torch.tensor([keys], device=q.device, dtype=torch.int32)
    scale = dim ** -0.5 if p.get('sm_scale') is None else p['sm_scale']

    def run(q, k, v, qs, ks, vs, cos, sin):
        q_scales = qs.repeat_interleave(repeat_kv_heads, dim=1)[:, :, None]
        q_scales = q_scales.expand(batch, heads, pages).contiguous()
        k_scales = ks.repeat_interleave(repeat_kv_heads, dim=1)[:, :, None].expand(batch, kvheads, pages).contiguous()
        v_scales = vs.repeat_interleave(repeat_kv_heads, dim=1)[:, :, None].expand(batch, kvheads, pages).contiguous()
        # Byte views preserve every FP8 encoding while using the native
        # repeat-interleave API on its supported uint8 datatype.
        output = torch.ops.classic_sgl_fp8.sparse_attention_score(
            q.contiguous().reshape(queries, heads, dim),
            k.contiguous().view(torch.uint8).repeat_interleave(repeat_kv_heads, dim=2).view(k.dtype).reshape(pages, page_size, kvheads, dim),
            v.contiguous().view(torch.uint8).repeat_interleave(repeat_kv_heads, dim=2).view(v.dtype).reshape(pages, page_size, kvheads, dim),
            selections, table, select_num_idx=valid, q_dequant_scale=q_scales,
            k_dequant_scale=k_scales, v_dequant_scale=v_scales,
            actual_seq_lengths=q_lengths, actual_seq_lengths_kv=kv_lengths,
            num_key_value_heads=kvheads, scale_value=scale, block_size=page_size,
            top_k=pages, inner_precise=0)
        return output.reshape(batch, queries, heads, dim)

    sources = [f for f in (root / 'csrc/sparse_attention_score').rglob('*')
               if f.suffix in ('.cpp', '.h', '.hpp')]
    return Candidate(Implementation(run), 'sgl_kernel_npu::npu_sparse_attention_score full-FP8',
                     'Repeat independent KV heads/scales to express GQA as MHA, retaining every original KV reduction; dense KV page views, all visible causal page indices, original partial diagonal-page mask, repeat supplied Q/K/V scale values into native block layouts. Wider index metadata extent preserves native logical Q-scale addressing. No external dequantization or native kernel/schedule changes.',
                     source_hashes(Path(__file__), bridge_source, Path(bridge_binary),
                                   root / 'csrc/utils/defines.h', binary, package / '__init__.py',
                                   root / 'csrc/pytorch_extensions.cpp', *sources),
                     limitations=('The original native kernel has16-entry page arrays; this focused case has14 pages. FP8 rounding must pass unchanged manifest tolerance. No success inferred from source correspondence.',))
