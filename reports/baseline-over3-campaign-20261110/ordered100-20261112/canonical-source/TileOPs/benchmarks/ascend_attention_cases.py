"""Shape/metadata adapters for original dense and sparse attention APIs."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes, vllm_native


def _contract(case):
    import torch

    q, k, v, *extra = case.inputs
    p = case.params
    if (any(t is not None for t in extra) or p.get('softcap') is not None
            or p.get('pos_encoding_mode', 'none') != 'none'
            or p.get('window_size_left', -1) != -1 or p.get('window_size_right', -1) != -1
            or q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype
            or p.get('out_dtype') not in (None, str(q.dtype).split('.')[-1])):
        raise UnsupportedCase('This adapter needs unscaled FP16/BF16 Q/K/V and full-context or causal attention; no added RoPE/softcap arithmetic.')
    batch, sq, heads, dim = q.shape
    sk, kvheads, vdim = k.shape[1], k.shape[2], v.shape[-1]
    if heads % kvheads or sq > sk or sk <= 0:
        raise UnsupportedCase('Native grouped attention needs integral head groups and nonempty visible prefixes.')
    scale = p.get('sm_scale')
    return batch, sq, heads, dim, sk, kvheads, vdim, (dim ** -0.5 if scale is None else scale)


def sgl_gqa_cases(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    batch, sq, heads, dim, sk, kvheads, vdim, scale = _contract(case)
    group = heads // kvheads
    padded_group = 1 << (group - 1).bit_length()
    page_size = 128
    pages = (sk + page_size - 1) // page_size
    table = torch.arange(batch * pages, device=case.inputs[0].device, dtype=torch.int32).view(batch, pages)
    lengths = [torch.full((batch,), sk - sq + i + 1 if case.params.get('is_causal') else sk,
                          device=case.inputs[0].device, dtype=torch.int32) for i in range(sq)]
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py'
    fn = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu', 'sgl_kernel_npu.attention.decode_attention', 'decode_gqa', host)
    def run(q, k, v, *extra):
        packed_q = F.pad(q.reshape(batch, sq, kvheads, group, dim), (0, 0, 0, padded_group - group))
        packed_q = packed_q.reshape(batch, sq, kvheads * padded_group, dim)
        packed_k = F.pad(k, (0, 0, 0, 0, 0, pages * page_size - sk)).contiguous().view(batch * pages, page_size, kvheads, dim)
        packed_v = F.pad(v, (0, 0, 0, 0, 0, pages * page_size - sk)).contiguous().view(batch * pages, page_size, kvheads, vdim)
        output = torch.empty((batch, sq, kvheads * padded_group, vdim), device=q.device, dtype=q.dtype)
        for i in range(sq):
            fn(packed_q[:, i].contiguous(), packed_k, packed_v, output[:, i], lengths[i], scale, page_size, table)
        return output.reshape(batch, sq, kvheads, padded_group, vdim)[..., :group, :].contiguous().reshape(batch, sq, heads, vdim)
    return Candidate(Implementation(run), 'sgl_kernel_npu.attention.decode_attention.decode_gqa',
                     'Pad independent query heads and final cache page; original length masks select each causal prefix; invoke original decode once per query; crop heads.',
                     source_hashes(Path(__file__), LIBRARIES / 'sgl-kernel-npu' / host),
                     limitations=('All per-call packing and per-query native launches count toward execution time.',))


def vllm_gqa_cases(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    batch, sq, heads, dim, sk, kvheads, vdim, scale = _contract(case)
    group = heads // kvheads
    if dim + vdim > 512 or group > 128:
        raise UnsupportedCase('Original sparse MLA needs K+V width<=512 and <=128 query heads per KV group.')
    padded_group = 128 if group > 64 else group
    ops, binary, vendor = vllm_native()
    tokens = torch.arange(sk, device=case.inputs[0].device, dtype=torch.int32).reshape(1, sk)
    visible = tokens.expand(sq, sk)
    if case.params.get('is_causal'):
        positions = torch.arange(sq, device=tokens.device, dtype=torch.int32) + sk - sq
        visible = torch.where(tokens <= positions[:, None], visible, -1)
    indices = visible.reshape(1, sq, 1, sk).expand(batch * kvheads, sq, 1, sk).contiguous()
    def run(q, k, v, *extra):
        queries = q.reshape(batch, sq, kvheads, group, dim).permute(0, 2, 1, 3, 4)
        packed_q = F.pad(queries, (0, 512 - dim, 0, padded_group - group)).reshape(batch * kvheads, sq, padded_group, 512)
        combined = torch.cat((k, v), dim=-1).permute(0, 2, 1, 3).contiguous()
        packed_kv = F.pad(combined, (0, 512 - dim - vdim)).reshape(batch * kvheads, sk, 1, 512)
        qr = torch.zeros((*packed_q.shape[:-1], 64), device=q.device, dtype=q.dtype)
        kr = torch.zeros((*packed_kv.shape[:-1], 64), device=q.device, dtype=q.dtype)
        output, _, _ = ops.npu_sparse_flash_attention(
            packed_q, packed_kv, packed_kv, indices, scale, query_rope=qr, key_rope=kr,
            sparse_block_size=1, layout_query='BSND', layout_kv='BSND', sparse_mode=0,
            attention_mode=2, return_softmax_lse=False)
        selected = output[..., :group, dim:dim + vdim].reshape(batch, kvheads, sq, group, vdim)
        return selected.permute(0, 2, 1, 3, 4).contiguous().reshape(batch, sq, heads, vdim)
    root = LIBRARIES / 'vllm-ascend'
    sources = [p for p in (root / 'csrc/attention/sparse_flash_attention').rglob('*') if p.suffix in ('.cpp', '.h', '.json')]
    deployed = [p for p in vendor.rglob('*') if p.is_file() and p.suffix in ('.so', '.o', '.json')]
    return Candidate(Implementation(run), 'vllm-ascend::npu_sparse_flash_attention',
                     'Fold KV groups into batch; native Q=[Q,0], KV=[K,V,0]; pad independent heads; causal indices use original -1 sentinel; select V output.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_baselines.py'),
                                   binary, root / 'csrc/torch_binding.cpp', *sources, *deployed),
                     limitations=('Original native sparse-attention launch attribution required.',))


def transformer_gqa_cases(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_transformer_graph import _candidate
    from dataclasses import replace

    batch, sq, heads, dim, sk, kvheads, vdim, scale = _contract(case)
    binding = GraphCall('ops-transformer')
    mask = None
    if case.params.get('is_causal') and sq > 1:
        q_positions = torch.arange(sq, device=case.inputs[0].device) + sk - sq
        k_positions = torch.arange(sk, device=case.inputs[0].device)
        mask = (k_positions[None, :] > q_positions[:, None]).contiguous()
    def run(q, k, v, *extra):
        out = torch.empty((batch, sq, heads, vdim), device=q.device, dtype=q.dtype)
        maximum = torch.empty((batch, heads, sq, 8), device=q.device, dtype=torch.float32)
        total = torch.empty_like(maximum)
        unused = torch.empty((0, 0, 0, 0), device=q.device, dtype=q.dtype)
        inputs = [q.contiguous(), k.contiguous(), v.contiguous(), *([None] * 16)]
        inputs[6] = mask  # Documented atten_mask input; True means invisible.
        binding.call('FlashAttentionScore', inputs, [maximum, total, unused, out], {
            'scale_value': float(scale), 'keep_prob': 1.0,
            'pre_tockens': 2147483647, 'next_tockens': 2147483647,
            'head_num': heads, 'input_layout': 'BSND', 'inner_precise': 0,
            'sparse_mode': 0,
        })
        return out
    native = _candidate(binding, run, 'FlashAttentionScore', 'attention/flash_attention_score',
                        'Original BSND grouped attention and optional documented mask input; right-aligned causal metadata for multi-query cases; unchanged native kernel/schedule.')
    return replace(native, source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
