"""Independent complete causal rows through the original sparse-attention API."""
from dataclasses import replace
from pathlib import Path

from benchmarks.ascend_baselines import source_hashes, vllm_native
from benchmarks.ascend_attention_cases import _contract, vllm_gqa_cases


def vllm_causal_rows(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    batch, sq, heads, dim, sk, kvheads, vdim, scale = _contract(case)
    base = vllm_gqa_cases(case)
    if sq == 1 or not case.params.get('is_causal'):
        return base
    ops, _, _ = vllm_native()
    group = heads // kvheads
    padded_group = 128 if group > 64 else group
    visible_lengths = list(range(sk - sq + 1, sk + 1))
    indices = [torch.arange(length, device=case.inputs[0].device, dtype=torch.int32)
               .reshape(1, 1, 1, length).expand(batch * kvheads, 1, 1, length).contiguous()
               for length in visible_lengths]

    def run(q, k, v, *extra):
        queries = q.reshape(batch, sq, kvheads, group, dim).permute(0, 2, 1, 3, 4)
        packed_q = F.pad(queries, (0, 512 - dim, 0, padded_group - group))
        packed_q = packed_q.reshape(batch * kvheads, sq, padded_group, 512)
        combined = torch.cat((k, v), dim=-1).permute(0, 2, 1, 3).contiguous()
        packed_kv = F.pad(combined, (0, 512 - dim - vdim)).reshape(batch * kvheads, sk, 1, 512)
        results = []
        for i, length in enumerate(visible_lengths):
            query = packed_q[:, i:i + 1].contiguous()
            kv = packed_kv[:, :length].contiguous()
            qr = torch.zeros((*query.shape[:-1], 64), device=q.device, dtype=q.dtype)
            kr = torch.zeros((*kv.shape[:-1], 64), device=q.device, dtype=q.dtype)
            output, _, _ = ops.npu_sparse_flash_attention(
                query, kv, kv, indices[i], scale, query_rope=qr, key_rope=kr,
                sparse_block_size=1, layout_query='BSND', layout_kv='BSND',
                sparse_mode=0, attention_mode=2, return_softmax_lse=False)
            results.append(output[..., :group, dim:dim + vdim])
        selected = torch.cat(results, dim=1).reshape(batch, kvheads, sq, group, vdim)
        return selected.permute(0, 2, 1, 3, 4).contiguous().reshape(batch, sq, heads, vdim)

    return replace(base, implementation=Implementation(run),
                   adaptation='Each independent query calls the unchanged original attention with its complete right-aligned visible KV prefix and all-valid indices. Full feature reductions stay native. Concatenate query rows; all per-call packing, slicing and native calls count in timing.',
                   source_sha256=base.source_sha256 | source_hashes(Path(__file__)),
                   limitations=(*base.limitations, 'Avoids invalid-index masking path implicated by077/092; root cause is not assumed proven. Full manifest verification required.'))


def vllm_rope_causal_cases(case):
    from benchmarks.ascend_rope_attention_cases import _with_rope
    return _with_rope(case, 'vllm-ascend', vllm_causal_rows)
