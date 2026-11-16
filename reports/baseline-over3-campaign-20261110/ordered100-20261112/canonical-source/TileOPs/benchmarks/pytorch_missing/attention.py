"""Paged cache operations and MLA/DSA using only device-side PyTorch tensors."""

import math

import torch

from benchmarks.pytorch_missing.pointwise import rotate


def _index_rows(source, indices):
    # Byte indexing supports FP8 on CPU and preserves all stored payload bits.
    if source.dtype == torch.float8_e4m3fn:
        return source.view(torch.uint8).index_select(0, indices).view(source.dtype)
    return source.index_select(0, indices)


def paged_cache_gather(
    dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None, *, out_dtype=None
):
    positions = torch.arange(dst.shape[0], device=dst.device)
    batch = torch.bucketize(positions, cu_seq_lens[1:].long(), right=True)
    logical = positions - cu_seq_lens.long()[batch]
    if seq_starts is not None:
        logical = logical + seq_starts.long()[batch]
    page = block_table[batch, logical // cache.shape[1]].long()
    slots = page * cache.shape[1] + logical % cache.shape[1]
    rows = _index_rows(cache.flatten(0, 1), slots)
    if scale is not None:
        rows = rows.float() * scale.float()
    dst.copy_(rows.to(dst.dtype))
    return dst


def mla_cache_write(
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    scale=None,
    positions=None,
    cos_sin_cache=None,
    *,
    fuse_rope=False,
    rope_layout="neox",
):
    pe = k_pe
    if fuse_rope:
        trig = cos_sin_cache.index_select(0, positions.long())
        pe = rotate(pe.float(), trig, rope_layout).to(k_pe.dtype)
    stored = torch.cat((kv_c, pe), -1)
    if kv_cache.dtype == torch.float8_e4m3fn:
        limit = torch.finfo(kv_cache.dtype).max
        stored = (stored.float() / scale.float()).clamp(-limit, limit)
    stored = stored.to(kv_cache.dtype).contiguous()
    flat = kv_cache.flatten(0, 1)
    # A unique scratch row for each skipped token avoids data-dependent nonzero
    # shapes, duplicate scatter destinations, and host reads of slot_mapping.
    scratch = torch.cat((flat.view(torch.uint8), torch.zeros_like(stored.view(torch.uint8))), 0)
    slots = slot_mapping.long()
    skipped = torch.arange(slots.numel(), device=slots.device) + flat.shape[0]
    slots = torch.where(slots >= 0, slots, skipped)
    scratch.index_copy_(0, slots, stored.view(torch.uint8))
    flat.view(torch.uint8).copy_(scratch[: flat.shape[0]])
    return kv_cache


def mla_paged(
    q,
    kv_cache,
    block_table,
    cache_seqlens,
    kv_scale=None,
    *,
    kv_lora_rank=512,
    is_causal=True,
    sm_scale=None,
):
    batch, queries, heads, width = q.shape
    capacity = block_table.shape[1] * kv_cache.shape[1]
    key_positions = torch.arange(capacity, device=q.device)
    scale = width**-0.5 if sm_scale is None else sm_scale
    outputs, lses = [], []
    for b in range(batch):
        pages = block_table[b].long()
        key = _index_rows(kv_cache, pages).reshape(capacity, width).float()
        if kv_scale is not None:
            key = key * kv_scale.float()
        parts, logs = [], []
        for start in range(0, queries, 16):
            end = min(start + 16, queries)
            scores = (q[b, start:end].float() @ key.T) * scale
            visible = key_positions < cache_seqlens[b]
            if is_causal:
                qp = cache_seqlens[b] - queries + torch.arange(start, end, device=q.device)
                visible = visible[None, :] & (key_positions[None, :] <= qp[:, None])
            else:
                visible = visible[None, :].expand(end - start, -1)
            scores = scores.masked_fill(~visible[:, None, :], -math.inf)
            logs.append(torch.logsumexp(scores, -1) / math.log(2))
            parts.append((torch.softmax(scores, -1) @ key[:, :kv_lora_rank]).to(q.dtype))
        outputs.append(torch.cat(parts, 0))
        lses.append(torch.cat(logs, 0))
    return torch.stack(outputs), torch.stack(lses)


def dsa_paged(q, kv_cache, block_table, cache_seqlens, indices, *, sm_scale=None):
    scale = q.shape[-1] ** -0.5 if sm_scale is None else sm_scale
    outputs, lses = [], []
    for b in range(q.shape[0]):
        positions = indices[b].long()
        valid = (positions >= 0) & (positions < cache_seqlens[b])
        safe = positions.clamp_min(0)
        pages = block_table[b].long()[safe // kv_cache.shape[1]]
        slots = pages * kv_cache.shape[1] + safe % kv_cache.shape[1]
        rows = kv_cache.flatten(0, 1).index_select(0, slots.flatten())
        codes = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float()
        scales = rows[:, 512:528].contiguous().view(torch.float32)
        latent = (codes.reshape(-1, 4, 128) * scales[..., None]).flatten(-2).to(torch.bfloat16)
        rope = rows[:, 528:].contiguous().view(torch.bfloat16)
        keys = torch.cat((latent, rope), -1).reshape(*positions.shape, 576).float()
        scores = (q[b].float() @ keys.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~valid[:, None, :], -math.inf)
        empty = ~valid.any(-1)
        safe_scores = torch.where(empty[:, None, None], torch.zeros_like(scores), scores)
        probability = torch.softmax(safe_scores, -1) * valid[:, None, :]
        outputs.append((probability @ keys[..., :512]).to(q.dtype))
        lse = torch.logsumexp(scores, -1)
        lses.append(torch.where(empty[:, None], torch.full_like(lse, math.inf), lse))
    return torch.stack(outputs), torch.stack(lses)
