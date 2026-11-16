"""Dense decode via original shared-KV sparse attention and concatenated K/V."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
from benchmarks.ascend_tla_baselines import _example_definitions


def tla_packed_gqa(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    q, k, v, *extra = case.inputs
    b, sq, hq, dim = q.shape
    sk, hk = k.shape[1:3]
    p = case.params
    if (sq != 1 or any(t is not None for t in extra) or k.shape != v.shape
            or q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype
            or hq % hk or sk % 64 or dim & (dim - 1)
            or p.get('softcap') is not None or p.get('pos_encoding_mode', 'none') != 'none'
            or p.get('window_size_left', -1) != -1 or p.get('window_size_right', -1) != -1):
        raise UnsupportedCase('Focused native alternative supports full-context FP16 decode, without RoPE/FP8/softcap.')
    group = hq // hk
    align = 16 if group <= 64 else 64
    padded_group = ((group + align - 1) // align) * align
    source = 'examples/sparse_flash_attention/example_sparse_flash_attn_gqa_pto.py'
    original = _example_definitions(source, 'sparse_attention_fwd')
    scale = p.get('sm_scale')
    scale = dim ** -0.5 if scale is None else float(scale)
    # Original dim is used for both key/value products. Packing Q'=[Q,0],
    # KV'=[K,V] makes Q'KV'^T=QK^T; select the V half of softmax(.)KV'.
    # All reductions, GEMMs, softmax, buffers and synchronization stay native.
    kernel = original(heads=hk * padded_group, dim=2 * dim, tail_dim=0, topk=sk,
                      workspace_block_num=b * sq * hk, kv_stride=1, kv_group=hk,
                      sm_scale=scale, is_causal=True)

    def run(q, k, v, *extra):
        query = q.reshape(b, sq, hk, group, dim)
        query = F.pad(query, (0, dim, 0, padded_group - group)).reshape(b, sq, hk * padded_group, 2 * dim)
        kv = torch.cat((k, v), dim=-1).contiguous()
        indices = torch.arange(sk, device=q.device, dtype=torch.int32).reshape(1, 1, 1, sk).expand(b, sq, hk, sk).contiguous()
        result = kernel(query, kv, indices).reshape(b, sq, hk, padded_group, 2 * dim)
        return result[:, :, :, :group, dim:].contiguous().reshape(q.shape)

    return Candidate(Implementation(run), 'tilelang-ascend::sparse_attention_fwd(shared-KV packed dense decode)',
                     'Concatenate K/V feature dimensions, zero-pad Q value half and unused heads; supply all KV indices and original manifest scale; select output V half. No arithmetic outside unchanged original kernel, no schedule changes.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_tla_baselines.py'),
                                   LIBRARIES / 'tilelang-ascend' / source),
                     limitations=('Distinct original PTO sparse kernel from quarantined varlenGQA. Frozen FP16 case must compile and pass before expansion; dense index/packing/extra output costs count in timing.',))
