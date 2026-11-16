"""Original FP8 FlashAttentionScore with native block-scale metadata."""
from dataclasses import replace
from pathlib import Path

from benchmarks.ascend_baselines import UnsupportedCase, source_hashes


def transformer_fp8_gqa_cases(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_transformer_graph import _candidate

    if case.inputs[0].dtype != torch.float8_e4m3fn:
        from benchmarks.ascend_rope_attention_cases import transformer_rope_gqa_cases
        return transformer_rope_gqa_cases(case)
    q, k, v, qs, ks, vs, cos, sin = case.inputs
    p = case.params
    batch, queries, heads, dim = q.shape
    keys, kvheads, value_dim = k.shape[1], k.shape[2], v.shape[-1]
    if (k.dtype != q.dtype or v.dtype != q.dtype or heads % kvheads or queries > keys
            or any(s is None or s.dtype != torch.float32 or tuple(s.shape) != (batch, kvheads)
                   for s in (qs, ks, vs))
            or cos is not None or sin is not None or p.get('pos_encoding_mode', 'none') != 'none'
            or p.get('softcap') is not None or p.get('out_dtype') != 'float16'
            or p.get('window_size_left', -1) != -1 or p.get('window_size_right', -1) != -1):
        raise UnsupportedCase('Prepared original FP8 full-context/causal entry with per-KV-head scales.')
    binding = GraphCall('ops-transformer')
    mask = None
    if p.get('is_causal'):
        q_positions = torch.arange(queries, device=q.device) + keys - queries
        k_positions = torch.arange(keys, device=q.device)
        mask = (k_positions[None, :] > q_positions[:, None]).contiguous()
    scale = dim ** -0.5 if p.get('sm_scale') is None else p['sm_scale']

    def run(q, k, v, qs, ks, vs, cos, sin):
        # Repeat scale values into the original Q128/K256/V512 block metadata.
        # No external dequantization, matmul, or softmax is performed here.
        dq = qs.repeat_interleave(heads // kvheads, dim=1)[:, :, None, None]
        dq = dq.expand(batch, heads, (queries + 127) // 128, 1).contiguous()
        dk = ks[:, :, None, None].expand(batch, kvheads, (keys + 255) // 256, 1).contiguous()
        dv = vs[:, :, None, None].expand(batch, kvheads, (keys + 511) // 512, 1).contiguous()
        maximum = torch.empty((batch, heads, queries, 8), dtype=torch.float32, device=q.device)
        total = torch.empty_like(maximum)
        # The ORIGINAL graph datatype inference fixes FP8 output to BF16.
        # Select its corresponding out_dtype=1, then convert the public output.
        unused = torch.empty((0, 0, 0, 0), dtype=torch.bfloat16, device=q.device)
        output = torch.empty((batch, queries, heads, value_dim), dtype=torch.bfloat16, device=q.device)
        inputs = [q.contiguous(), k.contiguous(), v.contiguous(), *([None] * 16)]
        inputs[6] = mask
        inputs[12:15] = [dq, dk, dv]
        binding.call('FlashAttentionScore', inputs, [maximum, total, unused, output], {
            'scale_value': float(scale), 'keep_prob': 1.0, 'head_num': heads,
            'pre_tockens': 2147483647, 'next_tockens': 2147483647,
            'input_layout': 'BSND', 'inner_precise': 0, 'sparse_mode': 0, 'out_dtype': 1,
        })
        return output.to(torch.float16)

    native = _candidate(binding, run, 'FlashAttentionScore', 'attention/flash_attention_score',
                        'Original FP8 kernel and independent native Q/K/V scale inputs, reshape/repeat per-head scales to documented blocks, native causal mask; native BF16 output cast to manifest FP16. No native source/schedule change.')
    return replace(native, source_sha256=native.source_sha256 | source_hashes(Path(__file__)),
                   limitations=(*native.limitations, 'Native FP8 probability and BF16 output rounding must pass unchanged manifest FP8 tolerance; no success is assumed from semantic correspondence.'))
