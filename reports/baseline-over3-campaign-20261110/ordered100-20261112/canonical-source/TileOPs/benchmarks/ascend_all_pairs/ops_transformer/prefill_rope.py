"""Staged original RoPE, cache append and full attention composition.

The original prefill oracle rounds the rotated new Q/K to storage dtype before
attention and cache append. Each original case requires its own native pilot.
"""
from pathlib import Path


def prefill_rope(case):
    import torch
    from dataclasses import replace
    from benchmarks.ascend_baselines import UnsupportedCase, WORKSPACE
    from benchmarks.ascend_graph_call import GraphCall
    from .native import GraphCall as CacheGraph
    from .adapters import _attention, _candidate
    from .hashing import source_hashes

    p = case.params
    q, kn, vn, kp, vp, *_ = case.inputs
    if case._call.signature.name != 'GQAPrefillPagedWithKVCacheFwdOp' or not p.get('fuse_rope'):
        raise UnsupportedCase('This composition binds original fused-RoPE prefill calls.')
    if p.get('softcap') or q.dtype not in (torch.float16, torch.bfloat16):
        raise UnsupportedCase('Original FP16/BF16 attention without softcap is required.')
    if any(t.dtype != q.dtype for t in (kn, vn, kp, vp)):
        raise UnsupportedCase('Scaled cache quantization requires a separate native mapping.')
    width = p.get('rotary_dim') or q.shape[-1]
    if width % 2 or width > q.shape[-1]:
        raise UnsupportedCase('Original even full/partial NeoX rotary width is required.')
    attention = GraphCall('ops-transformer')
    cache = CacheGraph('ops-transformer')

    def rotate(x, cos, sin):
        shape = (1, x.shape[0], x.shape[1], width)
        inp = x[..., :width].float().contiguous().reshape(shape)
        tables = [torch.cat((t, t), -1).float().reshape(1, x.shape[0], 1, width)
                  for t in (cos, sin)]
        output = torch.empty_like(inp)
        attention.call('RotaryPositionEmbedding', [inp, *tables, None], [output], {'mode': 0})
        rotated = output.reshape(*x.shape[:-1], width).to(x.dtype)
        return rotated if width == x.shape[-1] else torch.cat((rotated, x[..., width:]), -1)

    def run(q, kn, vn, kp, vp, ks, vs, cu, lens, table):
        bounds, lengths = cu.tolist(), lens.tolist()
        page = p['page_size']
        kc, vc, outputs = kp.clone(), vp.clone(), []
        for b, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            if end == begin:
                continue
            old = lengths[b]
            positions = torch.arange(end - begin, device=q.device) + old
            # Exactly the original prefill frequency formula and table cast.
            half = width // 2
            inverse = 1.0 / (p['rope_base'] **
                             (torch.arange(half, device=q.device, dtype=torch.float32) / half))
            angles = torch.outer(positions.float(), inverse)
            cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
            qr = rotate(q[begin:end], cos, sin)
            kr = rotate(kn[begin:end], cos, sin)
            all_positions = torch.arange(old + end - begin, device=q.device)
            rows = table[b, all_positions // page].long() * page + all_positions % page
            write = rows[old:].contiguous()
            k_view, v_view = [t.reshape(-1, page, *t.shape[1:]) for t in (kc, vc)]
            cache.call('ScatterPaKvCache',
                       [kr.contiguous(), k_view, write, vn[begin:end].contiguous(), v_view,
                        None, None, None], [k_view, v_view],
                       {'cache_mode': 'Norm', 'scatter_mode': 'None'})
            outputs.append(_attention(attention, qr[None], kc[rows][None], vc[rows][None], p)[0])
        return torch.cat(outputs), kc, vc

    result = _candidate(attention, run, 'RotaryPositionEmbedding+ScatterPaKvCache+FlashAttentionScore',
                        'attention/flash_attention_score',
                        'Original per-request RoPE positions and frequency-table casts, native FP32 rotation '
                        'with original storage rounding, private cache clones, original native append and '
                        'full attention. Complete attention/K/V outputs and preserved old/unused rows; '
                        'no quantization, kernel edit, tiling override or floating attention fallback.', host=True)

    def cleanup():
        attention.cleanup()
        cache.cleanup()

    return replace(result, cleanup=cleanup, source_sha256=result.source_sha256 | source_hashes(
        Path(__file__), cache.vendor,
        WORKSPACE / 'ascend_baselines/ops-transformer/posembedding/rotary_position_embedding',
        WORKSPACE / 'ascend_baselines/ops-transformer/attention/scatter_pa_kv_cache'))
