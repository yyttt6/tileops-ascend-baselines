"""Use original sparse MLA as dense GQA through padding and grouped layout."""
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, source_hashes, vllm_native,
)


def vllm_gqa(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    q, k, v, *extra = case.inputs
    params = case.params
    if (q.shape[1] != 1 or q.shape[-1] * 2 > 512 or any(t is not None for t in extra)
            or params.get("softcap") is not None
            or params.get("pos_encoding_mode", "none") != "none"
            or params.get("window_size_left", -1) != -1
            or params.get("window_size_right", -1) != -1):
        raise UnsupportedCase("This pilot is full-context, single-query GQA.")
    ops, binary, vendor = vllm_native()
    fn = ops.npu_sparse_flash_attention
    scale = params.get("sm_scale") or q.shape[-1] ** -0.5

    def run(q, k, v, *extra):
        batch, sq, heads, dim = q.shape
        sk, kvheads = k.shape[1:3]
        groups = heads // kvheads
        qpacked = q.reshape(batch, sq, kvheads, groups, dim).permute(0, 2, 1, 3, 4)
        qpacked = F.pad(qpacked.reshape(batch * kvheads, sq, groups, dim),
                        (0, 512 - dim)).contiguous()
        # The arch35 MLA kernel uses the key buffer for both K and V even
        # though its public API accepts a separate value argument. Encode
        # [K,V,zeros] in that shared buffer, and [Q,zeros] in the query.
        # QK still equals the original scores; the V channels of its output
        # are exactly the desired weighted sum.
        kv = torch.cat((k, v), dim=-1)
        packed = kv.permute(0, 2, 1, 3).reshape(batch * kvheads, sk, 1, 2 * dim)
        kpacked = F.pad(packed, (0, 512 - 2 * dim)).contiguous()
        indices = torch.arange(sk, device=q.device, dtype=torch.int32)
        indices = indices.reshape(1, 1, 1, sk).expand(batch * kvheads, sq, 1, sk).contiguous()
        # A5's mandatory rope component contributes exactly zero to QK.
        qr = torch.zeros((*qpacked.shape[:-1], 64), device=q.device, dtype=q.dtype)
        kr = torch.zeros((*kpacked.shape[:-1], 64), device=q.device, dtype=q.dtype)
        out, _, _ = fn(qpacked, kpacked, kpacked, indices, scale,
                       query_rope=qr, key_rope=kr, sparse_block_size=1,
                       layout_query="BSND", layout_kv="BSND",
                       sparse_mode=0, attention_mode=2, return_softmax_lse=False)
        return (out[..., dim:2 * dim].reshape(batch, kvheads, sq, groups, dim)
                .permute(0, 2, 1, 3, 4).reshape(q.shape).contiguous())

    root = LIBRARIES / "vllm-ascend"
    sources = [p for p in (root / "csrc/attention/sparse_flash_attention").rglob("*")
               if p.suffix in (".cpp", ".h", ".json")]
    deployed = [p for p in vendor.rglob("*")
                if p.is_file() and p.suffix in (".so", ".o", ".json")]
    return Candidate(
        Implementation(run), "vllm-ascend::npu_sparse_flash_attention",
        "Fold KV heads into batch; Q=[q,zeros], shared KV=[k,v,zeros] padded to512, zero rope64; "
        "all KV indices and original scale produce dense GQA; select V channels and unfold output.",
        source_hashes(Path(__file__), Path(__file__).with_name("ascend_baselines.py"),
                      binary, root / "csrc/torch_binding.cpp", *sources, *deployed),
        limitations=("Native sparse-attention kernel attribution is required.",),
    )
