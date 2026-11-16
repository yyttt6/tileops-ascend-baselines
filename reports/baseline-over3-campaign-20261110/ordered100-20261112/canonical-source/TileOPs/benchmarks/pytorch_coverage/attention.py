"""Device PyTorch implementations of the missing packed-attention contracts.

Only the small sequence-offset/length vectors are read on the host for loop
scheduling. Query, key, value, page-table and cache arithmetic stays on device.
FP8 movement uses byte views, avoiding unsupported FP8 index kernels without
changing any stored bits. No candidate calls the CPU references below.
"""

import math

import torch

OPS = {"MLAVarlenFwdOp", "GQAPrefillPagedWithKVCacheFwdOp", "PagedKVCacheWriteFwdOp"}
MUTATED = {"GQAPrefillPagedWithKVCacheFwdOp": (3, 4), "PagedKVCacheWriteFwdOp": (2, 3)}


def make_inputs(case, device):
    """Repair the observed GQA factory/manifest page-table disagreement only.

    The existing factory uses torch.randperm while declared metadata uses the
    manifest's seeded Python sample. Both describe disjoint random pages, but
    the shared runner correctly requires the declared values. Keep every
    original numerical tensor/distribution, replacing only this small table.
    Other operations retain their original inputs unchanged.
    """
    inputs = list(case.inputs)
    if case._call.signature.name == "GQAPrefillPagedWithKVCacheFwdOp":
        spec = case._call.specs["block_table"]
        inputs[9] = torch.tensor(spec.values, dtype=inputs[9].dtype, device=device).reshape(spec.shape)
    return tuple(inputs)


def _gather_rows(pages, rows):
    if pages.dtype == torch.float8_e4m3fn:
        return pages.view(torch.uint8).index_select(0, rows).view(pages.dtype)
    return pages.index_select(0, rows)


def _write_rows(pages, rows, values):
    if pages.dtype == torch.float8_e4m3fn:
        pages.view(torch.uint8).index_copy_(0, rows, values.contiguous().view(torch.uint8))
    else:
        pages.index_copy_(0, rows, values)


def _rope(x, positions, params):
    width = params["rotary_dim"] or x.shape[-1]
    half = width // 2
    freq = 1.0 / (params["rope_base"] ** (
        torch.arange(half, device=x.device, dtype=torch.float32) / half))
    angles = positions.float()[:, None] * freq[None, :]
    cosine = angles.cos().to(x.dtype).float()[:, None, :]
    sine = angles.sin().to(x.dtype).float()[:, None, :]
    first, second = x[..., :half].float(), x[..., half:width].float()
    rotated = torch.cat((first * cosine - second * sine,
                         second * cosine + first * sine), dim=-1).to(x.dtype)
    return torch.cat((rotated, x[..., width:]), dim=-1)


def _mla(params, q, k_nope, k_pe, v, cu_seqlens):
    scale = params["sm_scale"]
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    offsets = cu_seqlens.tolist()  # At most B+1 integer scheduling values.
    output = torch.empty((*q.shape[:2], v.shape[-1]), device=q.device, dtype=q.dtype)
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        length = end - begin
        for head in range(0, q.shape[1], 8):
            stop = min(head + 8, q.shape[1])
            keys = torch.cat((k_nope[begin:end, head:stop].float(),
                              k_pe[begin:end, None].float().expand(-1, stop-head, -1)), -1)
            keys = keys.permute(1, 2, 0).contiguous()
            values = v[begin:end, head:stop].transpose(0, 1).float().contiguous()
            for row in range(0, length, 512):
                last = min(row + 512, length)
                key_end = last if params["is_causal"] else length
                queries = q[begin+row:begin+last, head:stop].transpose(0, 1).float()
                scores = torch.matmul(queries, keys[..., :key_end]) * scale
                if params["is_causal"]:
                    visible = (torch.arange(key_end, device=q.device)[None, :]
                               <= torch.arange(row, last, device=q.device)[:, None])
                    scores.masked_fill_(~visible[None], -math.inf)
                probabilities = scores.softmax(-1)
                result = torch.matmul(probabilities, values[:, :key_end])
                output[begin+row:begin+last, head:stop] = result.transpose(0, 1).to(q.dtype)
                lse[begin+row:begin+last, head:stop] = scores.logsumexp(-1).transpose(0, 1)
    return output, lse


def _prefill(params, q, k_new, v_new, k_pages, v_pages, k_scale, v_scale,
             cu_seqlens_q, cache_seqlens, block_table):
    bounds = cu_seqlens_q.tolist()
    old_lengths = cache_seqlens.tolist()
    page_size = params["page_size"]
    scale = q.shape[-1] ** -0.5 if params["sm_scale"] is None else params["sm_scale"]
    quantized = k_pages.dtype == torch.float8_e4m3fn
    group = q.shape[1] // k_new.shape[1]
    output = torch.empty_like(q)
    appends = []
    for batch, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        length, old_length = end-begin, old_lengths[batch]
        positions = torch.arange(old_length, device=q.device)
        rows = block_table[batch].long()[positions // page_size] * page_size + positions % page_size
        old_k, old_v = _gather_rows(k_pages, rows), _gather_rows(v_pages, rows)
        if quantized:
            old_k = (old_k.float() * k_scale[0]).to(q.dtype)
            old_v = (old_v.float() * v_scale[0]).to(q.dtype)
        query, key = q[begin:end], k_new[begin:end]
        q_positions = torch.arange(length, device=q.device) + old_length
        if params["fuse_rope"]:
            query, key = _rope(query, q_positions, params), _rope(key, q_positions, params)
        new_rows = (block_table[batch].long()[q_positions // page_size] * page_size
                    + q_positions % page_size)
        appends.append((new_rows, key, v_new[begin:end]))
        all_k = torch.cat((old_k, key))
        all_v = torch.cat((old_v, v_new[begin:end]))
        for head in range(0, q.shape[1], 8):
            last_head = min(head + 8, q.shape[1])
            kv_heads = torch.arange(head, last_head, device=q.device) // group
            keys = all_k.index_select(1, kv_heads).permute(1, 2, 0).float().contiguous()
            values = all_v.index_select(1, kv_heads).transpose(0, 1).float().contiguous()
            for row in range(0, length, 256):
                last = min(row + 256, length)
                scores = torch.matmul(query[row:last, head:last_head].transpose(0, 1).float(), keys)
                scores = scores * scale
                if params["softcap"] is not None and params["softcap"] > 0:
                    scores = params["softcap"] * torch.tanh(scores / params["softcap"])
                if params["is_causal"]:
                    visible = (torch.arange(old_length+length, device=q.device)[None, :]
                               <= q_positions[row:last, None])
                    scores.masked_fill_(~visible[None], -math.inf)
                result = torch.matmul(scores.softmax(-1).nan_to_num(), values)
                output[begin+row:begin+last, head:last_head] = result.transpose(0, 1).to(q.dtype)
    # Read all old pages before any append, matching the reference with aliases.
    for rows, key, value in appends:
        if quantized:
            # CPU PyTorch's FP8 conversion saturates overflow; the NPU cast
            # emits NaN instead. Express that reference conversion explicitly,
            # while preserving NaN inputs and every untouched cache byte.
            limit = torch.finfo(k_pages.dtype).max
            key = (key.float() / k_scale[0]).clamp(-limit, limit).to(k_pages.dtype)
            value = (value.float() / v_scale[0]).clamp(-limit, limit).to(v_pages.dtype)
        _write_rows(k_pages, rows, key)
        _write_rows(v_pages, rows, value)
    return output, k_pages, v_pages


def _cache_write(k, v, k_pages, v_pages, slots, k_scale=None, v_scale=None):
    written = (slots >= 0).nonzero().flatten()
    rows = slots.index_select(0, written)
    for pages, source, scale in ((k_pages, k, k_scale), (v_pages, v, v_scale)):
        values = source.index_select(0, written)
        if scale is not None:
            limit = torch.finfo(pages.dtype).max
            values = (values.float() / scale[0]).clamp(-limit, limit).to(pages.dtype)
        _write_rows(pages.view(-1, *pages.shape[2:]), rows, values)
    return k_pages, v_pages


def bind(case):
    op = case._call.signature.name
    params = dict(case._call.params)
    if op == "MLAVarlenFwdOp":
        return lambda *inputs: _mla(params, *inputs)
    if op == "GQAPrefillPagedWithKVCacheFwdOp":
        return lambda *inputs: _prefill(params, *inputs)
    if op == "PagedKVCacheWriteFwdOp":
        return _cache_write
    raise KeyError(op)


def cpu_reference(case, inputs):
    """Independent CPU path, with bounded GQA score slabs and FP8-safe indexing.

    MLA uses its existing independent chunked CPU oracle. GQA gathers pages as
    float32 on CPU, and computes each KV group independently; cache writes use
    byte assignment after CPU conversion. Never called by a device candidate.
    """
    assert all(not isinstance(t, torch.Tensor) or t.device.type == "cpu" for t in inputs)
    op = case._call.signature.name
    if op == "MLAVarlenFwdOp":
        return case.reference(*inputs)
    if op == "PagedKVCacheWriteFwdOp":
        k, v, kp, vp, slots, ks, vs = inputs
        outputs = []
        for original, source, scale in ((kp, k, ks), (vp, v, vs)):
            result = original.clone()
            valid = slots >= 0
            values = source[valid].float()
            if scale is not None:
                limit = torch.finfo(original.dtype).max
                values = (values / scale[0]).clamp(-limit, limit)
            flat = result.view(-1, *original.shape[2:]).view(torch.uint8)
            flat[slots[valid]] = values.to(original.dtype).contiguous().view(torch.uint8)
            outputs.append(result)
        return tuple(outputs)
    if op != "GQAPrefillPagedWithKVCacheFwdOp":
        raise KeyError(op)
    q, k, v, kp, vp, ks, vs, cu, lengths, table = inputs
    params = case._call.params
    page_size = params["page_size"]
    scale = q.shape[-1] ** -0.5 if params["sm_scale"] is None else params["sm_scale"]
    quantized = kp.dtype == torch.float8_e4m3fn
    result = torch.empty_like(q)
    kp_result, vp_result = kp.clone(), vp.clone()
    appends = []
    for b, (begin, end) in enumerate(zip(cu.tolist()[:-1], cu.tolist()[1:])):
        old_length = int(lengths[b])
        old_keys, old_values = [], []
        for offset in range(0, old_length, page_size):
            physical = int(table[b, offset // page_size]) * page_size
            count = min(page_size, old_length-offset)
            old_keys.append(kp[physical:physical+count].float())
            old_values.append(vp[physical:physical+count].float())
        shape = (0, *k.shape[1:])
        old_k = torch.cat(old_keys) if old_keys else torch.empty(shape)
        old_v = torch.cat(old_values) if old_values else torch.empty(shape)
        if quantized:
            old_k, old_v = old_k * ks[0], old_v * vs[0]
        old_k, old_v = old_k.to(q.dtype), old_v.to(q.dtype)
        query, key = q[begin:end], k[begin:end]
        positions = torch.arange(end-begin) + old_length
        if params["fuse_rope"]:
            query, key = case.workload.rope(query, positions), case.workload.rope(key, positions)
        keys, values = torch.cat((old_k, key)), torch.cat((old_v, v[begin:end]))
        group = q.shape[1] // k.shape[1]
        for kv_head in range(k.shape[1]):
            lo, hi = kv_head*group, (kv_head+1)*group
            for row in range(0, end-begin, 128):
                last = min(row+128, end-begin)
                scores = torch.einsum("rhd,kd->hrk", query[row:last, lo:hi].float(),
                                      keys[:, kv_head].float()) * scale
                if params["softcap"] is not None and params["softcap"] > 0:
                    scores = scores.div(params["softcap"]).tanh() * params["softcap"]
                if params["is_causal"]:
                    mask = torch.arange(keys.shape[0])[None, :] > positions[row:last, None]
                    scores.masked_fill_(mask[None], -math.inf)
                out = torch.einsum("hrk,kd->rhd", scores.softmax(-1).nan_to_num(), values[:, kv_head].float())
                result[begin+row:begin+last, lo:hi] = out.to(q.dtype)
        rows = table[b, positions // page_size].long() * page_size + positions % page_size
        appends.append((rows, key, v[begin:end]))
    for rows, key, value in appends:
        if quantized:
            key, value = (key.float()/ks[0]).to(kp.dtype), (value.float()/vs[0]).to(vp.dtype)
        kp_result.view(torch.uint8)[rows] = key.contiguous().view(torch.uint8)
        vp_result.view(torch.uint8)[rows] = value.contiguous().view(torch.uint8)
    return result, kp_result, vp_result
