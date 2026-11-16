"""Staged original RoPE plus packed FP16/BF16 attention composition.

The manifest Varlen reference rounds each FP32 rotation to input dtype before
the score product. Paged GQA has a different precision contract and is excluded.
Every original case requires fresh complete-output and native ownership evidence.
"""
from pathlib import Path


def varlen_rope(case):
    import torch
    from dataclasses import replace
    from benchmarks.ascend_baselines import LIBRARIES, UnsupportedCase
    from benchmarks.ascend_graph_call import GraphCall
    from .adapters import _attention, _candidate
    from .hashing import source_hashes

    p = case.params
    q, k, v, cq, ck, *extra = case.inputs
    slots = list(extra) + [None] * (6 - len(extra))
    q_scale, k_scale, v_scale, cos, sin, sinks = slots
    if case._call.signature.name != 'GQAVarlenFwdOp' or p.get('pos_encoding_mode') != 'rope':
        raise UnsupportedCase('This composition binds only original Varlen RoPE attention.')
    if any(x is not None for x in (q_scale, k_scale, v_scale, sinks)) or p.get('softcap') is not None:
        raise UnsupportedCase('Quantization, sinks and softcap require separate native mappings.')
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise UnsupportedCase('Original attention input pointers require matching FP16/BF16.')
    if p.get('out_dtype') not in (None, str(q.dtype).removeprefix('torch.')):
        raise UnsupportedCase('This route preserves the original Varlen input/output precision.')
    if cos is None or sin is None:
        raise UnsupportedCase('Original NPU-generated RoPE tables are required.')
    width = p.get('rotary_dim') or q.shape[-1]
    layout = p.get('rope_layout', 'neox')
    if width % 2 or width > q.shape[-1] or layout not in ('neox', 'interleaved'):
        raise UnsupportedCase('Original even full/partial NeoX/interleaved rotation is required.')
    if cos.shape != sin.shape or cos.shape[-1] != width // 2:
        raise UnsupportedCase('Original table width must equal half the declared rotary width.')
    binding = GraphCall('ops-transformer')

    def rotate(x, start, stop, cos, sin):
        shape = (1, x.shape[0], x.shape[1], width)
        packed = x[..., :width].float().contiguous().reshape(shape)
        if layout == 'interleaved':
            packed = packed.reshape(*shape[:-1], width // 2, 2).transpose(-1, -2).contiguous().reshape(shape)
        tables = [torch.cat((t[start:stop], t[start:stop]), -1).float().reshape(1, x.shape[0], 1, width)
                  for t in (cos, sin)]
        output = torch.empty_like(packed)
        binding.call('RotaryPositionEmbedding', [packed, *tables, None], [output], {'mode': 0})
        output = output.reshape(*x.shape[:-1], width)
        if layout == 'interleaved':
            output = output.reshape(*output.shape[:-1], 2, width // 2).transpose(-1, -2).contiguous().reshape(*x.shape[:-1], width)
        # The original Varlen oracle performs this exact storage conversion.
        output = output.to(x.dtype)
        return output if width == x.shape[-1] else torch.cat((output, x[..., width:]), -1)

    def run(q, k, v, cq, ck, q_scale=None, k_scale=None, v_scale=None,
            rope_cos=None, rope_sin=None, sinks=None):
        qb, kb = cq.tolist(), ck.tolist()
        outputs = []
        for i in range(len(qb) - 1):
            qlen, klen = qb[i + 1] - qb[i], kb[i + 1] - kb[i]
            if qlen == 0:
                continue  # Original ragged cases contain empty segments; no outputs belong to them.
            if qlen > klen or klen == 0:
                raise UnsupportedCase('Nonempty queries require right-aligned nonempty key segments.')
            qr = rotate(q[qb[i]:qb[i + 1]], klen - qlen, klen, rope_cos, rope_sin)
            kr = rotate(k[kb[i]:kb[i + 1]], 0, klen, rope_cos, rope_sin)
            outputs.append(_attention(binding, qr[None], kr[None], v[kb[i]:kb[i + 1]][None], p)[0])
        return torch.cat(outputs).contiguous()

    result = _candidate(binding, run, 'FlashAttentionScore+RotaryPositionEmbedding',
                        'attention/flash_attention_score',
                        'Original full per-sequence RoPE tables and right-aligned positions; exact FP32 widening, '
                        'native mode-0 rotation with optional interleaved packing, manifest storage rounding, '
                        'then original complete native attention. All copies and metadata access remain in run.', host=True)
    rope_root = LIBRARIES / 'ops-transformer/posembedding/rotary_position_embedding'
    files = [f for f in rope_root.rglob('*') if f.is_file() and f.suffix in ('.cpp', '.h', '.py', '.json')]
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__), *files))
