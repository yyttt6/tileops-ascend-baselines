"""Layout adapters to the unchanged FlashMLA Ascend sparse-prefill API.

All floating-point math is performed by FlashMLA. Torch use below is limited to
allocation, copies, views, indexing/padding and output dtype conversion. Arithmetic
on integer/bool sequence/window metadata constructs the original attention mask.
No input-derived tensor or metadata is cached across calls.
"""
from math import prod
from .contract import classify, axes_plan, pool_window
from .native import load_native


def _prepare(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    op, p = case._call.signature.name, case.params
    supported, reason = classify(op, p)
    if not supported:
        raise UnsupportedCase(reason)
    native, hashes = load_native()
    return op, p, native, hashes


def _manifest_guard(op, p, run):
    """Refuse shortened workload tensors or partial outputs, without tensor math."""
    names = {
        "GQADenseFwdOp": ("q", "k", "v", "q_scale", "k_scale", "v_scale", "rope_cos", "rope_sin"),
        "GQAVarlenFwdOp": ("q", "k", "v", "cu_seqlens_q", "cu_seqlens_kv", "q_scale", "k_scale", "v_scale", "rope_cos", "rope_sin", "sinks"),
        "GQAPagedFwdOp": ("q", "k_pages", "v_pages", "page_table", "cache_seqlens", "cu_seqlens_q", "q_scale", "k_scale", "v_scale", "rope_cos", "rope_sin", "sinks"),
        "GQAPrefillPagedWithKVCacheFwdOp": ("q", "k_new", "v_new", "k_pages", "v_pages", "k_scale", "v_scale", "cu_seqlens_q", "cache_seqlens", "block_table"),
        "DSADecodeWithKVCacheFwdOp": ("q", "kv", "indices"),
        "NSAVarlenFwdOp": ("q", "k", "v", "block_indices", "block_counts", "offsets", "token_indices"),
        "NSACompressedVarlenFwdOp": ("q", "k_cmp", "v_cmp", "offsets", "chunk_offsets", "token_indices"),
        "MLAVarlenFwdOp": ("q", "k_nope", "k_pe", "v", "cu_seqlens"),
        "MeanPoolingFwdOp": ("x", "offsets", "indices"),
        "BmmFwdOp": ("a", "b"),
        "MaximumFwdOp": ("input", "other"),
        "ReluFwdOp": ("input",),
    }.get(op, ("input",) if "Pool" in op else ("x",))
    if op == "BmmFwdOp":
        outputs = ("d",)
    elif op == "GQAPrefillPagedWithKVCacheFwdOp":
        outputs = ("o", "k_pages", "v_pages")
    elif op in {"NSACompressedVarlenFwdOp", "MLAVarlenFwdOp"}:
        outputs = ("o", "lse")
    elif op == "NSAVarlenFwdOp":
        outputs = ("o_slc",)
    else:
        outputs = ("o",) if "o" in p else ("output",)

    def check(value, name):
        shape, dtype = p[name]
        if value is None or tuple(value.shape) != tuple(shape) or str(value.dtype).removeprefix("torch.") != dtype:
            raise ValueError(f"{op} manifest tensor {name} requires shape={tuple(shape)}, dtype={dtype}; workload/candidate contract mismatch")

    def checked(*args):
        if len(args) > len(names):
            raise ValueError(f"{op}: unexpected input slots")
        for i, name in enumerate(names):
            value = args[i] if i < len(args) else None
            if name in p:
                check(value, name)
            elif value is not None:
                raise ValueError(f"{op}: undeclared input {name}")
        result = run(*args)
        values = result if isinstance(result, tuple) else (result,)
        if len(values) != len(outputs):
            raise ValueError(f"{op}: incomplete manifest outputs")
        for value, name in zip(values, outputs):
            check(value, name)
        return result
    return checked


def _candidate(run, hashes, adaptation, *, op, params, host=False, statistic="output"):
    from benchmarks.ascend_baselines import Candidate
    from benchmarks.api import Implementation
    return Candidate(
        Implementation(_manifest_guard(op, params, run)), "FlashMLA::sparse_prefill_fwd", f"Select {statistic}. {adaptation}", hashes,
        limitations=("CPU source/compile review only; no numerical or NPU validation in this task.",
                     "Full native attention executes even when only a statistic is selected; all per-call packing is timed.",
                     "Fixed native H=64,D=512 can be expensive for reductions/pooling; bounded host batches limit workspace, not kernel schedule."),
        timing_mode="host_roundtrip" if host else "device_async",
    )


def _visible_indices(start, stop, qlen, klen, p, device):
    import torch
    count = max(64, (klen + 63) // 64 * 64)
    key = torch.arange(count, dtype=torch.int32, device=device)[None, :]
    query = torch.arange(start, stop, dtype=torch.int32, device=device)[:, None] + klen - qlen
    visible = (key < klen).expand(stop - start, -1)
    if p.get("is_causal", False):
        visible = visible & (key <= query)
    if p.get("window_size_left", -1) >= 0:
        visible = visible & (key >= query - p["window_size_left"])
    if p.get("window_size_right", -1) >= 0:
        visible = visible & (key <= query + p["window_size_right"])
    return torch.where(visible, key, -1).contiguous()


def _attend(native, q, k, v, p, *, indices=None, sinks=None, shared_prefix=False):
    """One packed request. indices(start,stop,kvhead) returns local token IDs."""
    import torch
    import torch.nn.functional as F
    nt, heads, dim = q.shape
    sk, groups = k.shape[:2]
    vdim, group = v.shape[-1], heads // groups
    scale = p.get("sm_scale", p.get("scale"))
    scale = dim ** -0.5 if scale is None else scale
    out = torch.empty((nt, heads, vdim), dtype=q.dtype, device=q.device)
    lse = torch.empty((nt, heads), dtype=torch.float32, device=q.device)
    for g in range(groups):
        values = k[:, g] if shared_prefix else torch.cat((k[:, g], v[:, g]), dim=-1)
        kv = F.pad(values.contiguous(), (0, 512 - values.shape[-1])).reshape(sk, 1, 512)
        vstart = 0 if shared_prefix else dim
        for h in range(0, group, 64):
            begin, nh = g * group + h, min(64, group - h)
            sink = None if sinks is None else F.pad(sinks[begin:begin + nh].contiguous(), (0, 64 - nh), value=-float("inf"))
            for start in range(0, nt, 64):
                stop = min(nt, start + 64)
                query = F.pad(q[start:stop, begin:begin + nh].contiguous(), (0, 512 - dim, 0, 64 - nh))
                idx = (_visible_indices(start, stop, nt, sk, p, q.device)
                       if indices is None else indices(start, stop, g))
                idx = F.pad(idx.to(torch.int32), (0, (-idx.shape[-1]) % 64), value=-1)
                if idx.shape[-1] == 0:
                    idx = torch.full((stop - start, 64), -1, dtype=torch.int32, device=q.device)
                result, _, stats = native(query, kv, idx.contiguous().view(stop - start, 1, -1), scale, 512, sink, None)
                out[start:stop, begin:begin + nh] = result[:, :nh, vstart:vstart + vdim]
                lse[start:stop, begin:begin + nh] = stats[:, :nh]
    return out, lse


def _reject_extras(values):
    from benchmarks.ascend_baselines import UnsupportedCase
    if any(value is not None for value in values):
        raise UnsupportedCase("Unexpected runtime scale/RoPE input; no input may be silently ignored")


def attention(case):
    op, p, native, hashes = _prepare(case)

    def run(*args):
        import torch
        if op == "GQADenseFwdOp":
            q, k, v, *extra = args
            _reject_extras(extra)
            return torch.stack([_attend(native, q[b], k[b], v[b], p)[0] for b in range(q.shape[0])])
        if op == "GQAVarlenFwdOp":
            q, k, v, cuq, cuk, *extra = args
            extra = [*extra, *([None] * (6 - len(extra)))]
            _reject_extras(extra[:5])
            sinks = extra[5]
            qb, kb = cuq.tolist(), cuk.tolist()
            return torch.cat([_attend(native, q[qb[b]:qb[b + 1]], k[kb[b]:kb[b + 1]], v[kb[b]:kb[b + 1]], p, sinks=sinks)[0]
                              for b in range(len(qb) - 1)])
        if op == "MLAVarlenFwdOp":
            q, knope, kpe, v, offsets = args
            # k_pe is already encoded by the caller; this is concatenation only.
            k = torch.cat((knope, kpe[:, None, :].expand(-1, knope.shape[1], -1)), dim=-1)
            bounds, outputs, stats = offsets.tolist(), [], []
            for lo, hi in zip(bounds, bounds[1:]):
                out, lse = _attend(native, q[lo:hi], k[lo:hi], v[lo:hi], p)
                outputs.append(out)
                stats.append(lse)
            return torch.cat(outputs), torch.cat(stats)
        if op == "GQAPagedFwdOp":
            q, kpages, vpages, table, lengths, cuq, *extra = args
            extra = [*extra, *([None] * (6 - len(extra)))]
            _reject_extras(extra[:5])
            sinks = extra[5]
            qb, lens, page = cuq.tolist(), lengths.tolist(), kpages.shape[1]
            outputs = []
            for b, length in enumerate(lens):
                pages = table[b, :(length + page - 1) // page].long()
                k, v = (x[pages].flatten(0, 1)[:length] for x in (kpages, vpages))
                outputs.append(_attend(native, q[qb[b]:qb[b + 1]], k, v, p, sinks=sinks)[0])
            return torch.cat(outputs)
        if op == "GQAPrefillPagedWithKVCacheFwdOp":
            q, knew, vnew, kpages, vpages, _kscale, _vscale, cuq, lengths, table = args
            # BF16 caches ignore scale slots in the original contract. Copy-on-write
            # keeps all inputs intact and returns BOTH complete updated cache arrays.
            kp, vp = kpages.clone(), vpages.clone()
            qb, lens, page = cuq.tolist(), lengths.tolist(), p["page_size"]
            outputs, writes = [], []
            for b, old in enumerate(lens):
                lo, hi = qb[b:b + 2]
                oldpos = torch.arange(old, device=q.device)
                rows = table[b, oldpos // page].long() * page + oldpos % page
                k = torch.cat((kpages[rows], knew[lo:hi]))
                v = torch.cat((vpages[rows], vnew[lo:hi]))
                outputs.append(_attend(native, q[lo:hi], k, v, p)[0])
                newpos = torch.arange(hi - lo, device=q.device) + old
                newrows = table[b, newpos // page].long() * page + newpos % page
                writes.append((newrows, knew[lo:hi], vnew[lo:hi]))
            for rows, nk, nv in writes:
                kp[rows], vp[rows] = nk, nv
            return torch.cat(outputs), kp, vp
        if op == "DSADecodeWithKVCacheFwdOp":
            q, kv, selection = args
            sq, sk = q.shape[1], kv.shape[1]
            stride = p["stride_kv"]
            qbase = p.get("q_start_index_s")
            qbase = sk * stride - sq if qbase is None else qbase
            outputs = []
            for b in range(q.shape[0]):
                def selected(start, stop, g):
                    # The oracle uses a boolean selection mask: duplicates count once.
                    idx = selection[b, start:stop, g].long()
                    pos = torch.arange(start, stop, device=q.device)[:, None]
                    valid = (idx >= 0) & (idx < sk) & (idx * stride + stride - 1 <= pos + qbase)
                    idx = torch.where(valid, idx, -1)
                    forced = torch.where(pos < stride - 1, 0, -1)
                    idx = torch.cat((idx, forced), dim=1).sort(dim=-1).values
                    unique = torch.ones_like(idx, dtype=torch.bool)
                    unique[:, 1:] = idx[:, 1:] != idx[:, :-1]
                    return torch.where(unique, idx, -1)
                output, _ = _attend(native, q[b], kv[b], kv[b, ..., :p["o"][0][-1]], p,
                                    indices=selected, shared_prefix=True)
                outputs.append(output)
            return torch.stack(outputs)
        if op == "NSAVarlenFwdOp":
            q, k, v, blocks, counts, offsets, _token_indices = args
            bounds, bs, outputs = offsets.tolist(), p["block_size"], []
            for lo, hi in zip(bounds, bounds[1:]):
                def selected(start, stop, g):
                    block = blocks[lo + start:lo + stop, g].long()
                    pos = (block[..., None] * bs + torch.arange(bs, device=q.device)).flatten(1)
                    slot = torch.arange(block.shape[1], device=q.device).repeat_interleave(bs)[None, :]
                    valid = (pos >= 0) & (pos < hi - lo) & (slot < counts[lo + start:lo + stop, g, None])
                    if p["is_causal"]:
                        valid = valid & (pos <= torch.arange(start, stop, device=q.device)[:, None])
                    return torch.where(valid, pos, -1)
                outputs.append(_attend(native, q[lo:hi], k[lo:hi], v[lo:hi], p, indices=selected)[0])
            return torch.cat(outputs)
        if op == "NSACompressedVarlenFwdOp":
            q, k, v, offsets, chunk_offsets, _token_indices = args
            bounds, cb, bs = offsets.tolist(), chunk_offsets.tolist(), p["bs"]
            outputs, statistics = [], []
            for b, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
                # The reference defines zero output AND zero LSE until the first
                # compression block closes. Represent these as output padding.
                prefix = min(bs - 1, hi - lo)
                pre_o = q.new_zeros((prefix, q.shape[1], v.shape[-1]))
                pre_l = q.new_zeros((prefix, q.shape[1]))
                nk = (hi - lo) // bs
                if nk == 0:
                    outputs.append(pre_o)
                    statistics.append(pre_l)
                    continue
                def selected(start, stop, g):
                    token = torch.arange(start + prefix, stop + prefix, device=q.device)[:, None]
                    key = torch.arange(nk, device=q.device)[None, :]
                    return torch.where(key < (token + 1) // bs, key, -1)
                out, lse = _attend(native, q[lo + prefix:hi], k[cb[b]:cb[b] + nk], v[cb[b]:cb[b] + nk], p, indices=selected)
                outputs.append(torch.cat((pre_o, out)))
                statistics.append(torch.cat((pre_l, lse.to(q.dtype))))
            return torch.cat(outputs), torch.cat(statistics)
        raise AssertionError(op)

    return _candidate(run, hashes,
                      "Original sparse prefill; pack separate K/V or preserve shared KV prefix, pad heads/width/topk, map causal/window/paged/block indices and native sinks; return every declared output, including full cloned caches or LSE.",
                      op=op, params=p, host=op not in {"GQADenseFwdOp", "DSADecodeWithKVCacheFwdOp"},
                      statistic="output,lse" if op in {"NSACompressedVarlenFwdOp", "MLAVarlenFwdOp"} else "output")


def _reduce_rows(native, rows, mode, valid=None):
    """Pack independent rows into native columns; valid is a shared slot mask.

    For max/LSE, basis query head h selects KV column h. For mean, zero Q
    averages all packed columns simultaneously. This is the representative
    scalar basis-query layout batched across the existing native heads.
    """
    import torch
    import torch.nn.functional as F
    nrow, width = rows.shape
    outputs = []
    slots = 512 if mode == "mean" else 64
    # At least one complete domain is required; very wide rows can exceed32MiB.
    group_batch = max(1, min(64, (32 * 1024**2) // (max(1, width) * 512 * 2)))
    batch = slots * group_batch
    for start in range(0, nrow, batch):
        stop = min(nrow, start + batch)
        nr = stop - start
        groups = (nr + slots - 1) // slots
        packed = F.pad(rows[start:stop], (0, 0, 0, groups * slots - nr))
        packed = packed.reshape(groups, slots, width).transpose(1, 2).contiguous()
        kv = F.pad(packed, (0, 512 - slots)).reshape(groups * width, 1, 512)
        q = torch.zeros((groups, 64, 512), dtype=rows.dtype, device=rows.device)
        if mode != "mean":
            diagonal = torch.arange(slots, device=rows.device)
            q[:, diagonal, diagonal] = 1
        columns = torch.arange((width + 63) // 64 * 64, device=rows.device, dtype=torch.int32)[None, :]
        visible = columns < width
        if valid is not None:
            visible = visible & F.pad(valid, (0, columns.shape[1] - width), value=False)
        offsets = torch.arange(groups, dtype=torch.int32, device=rows.device)[:, None] * width
        idx = torch.where(visible, columns + offsets, -1).view(groups, 1, -1)
        out, maximum, lse = native(q, kv, idx, 1.0, 512, None, None)
        value = out[:, 0, :slots] if mode == "mean" else (maximum if mode == "max" else lse)
        outputs.append(value.reshape(-1)[:nr].to(rows.dtype))
    return torch.cat(outputs)


def reduction(case):
    op, p, native, hashes = _prepare(case)
    remaining, axes, width, output_shape = axes_plan(p["x"][0], p.get("dim"), p.get("keepdim", False))
    mode = {"AmaxFwdOp": "max", "LogSumExpFwdOp": "lse", "MeanFwdOp": "mean", "AnyFwdOp": "max"}[op]

    def run(x):
        import torch
        rows = x.permute(*remaining, *axes).contiguous().reshape(-1, width) if x.ndim else x.reshape(1, 1)
        if op == "AnyFwdOp":
            rows = rows.to(torch.bfloat16)  # boolean0/1 is represented exactly
        return _reduce_rows(native, rows, mode).to(x.dtype).reshape(output_shape)

    return _candidate(run, hashes,
                      "Permute reduction axes; zero-pad scalar K/V. Basis Q exposes max_logits or natural-log LSE; zero Q gives native uniform weighted mean. Select statistic and cast to manifest output dtype.",
                      op=op, params=p, statistic=mode)


def pool(case):
    op, p, native, hashes = _prepare(case)
    mode = "max" if "Max" in op else "mean"
    input_shape, output_shape = tuple(p["input"][0]), tuple(p["output"][0])
    spatial, outspatial = input_shape[2:], output_shape[2:]

    def run(x):
        import torch
        from itertools import product
        # Only layout/index generation uses host arithmetic. All value reductions
        # call the original native kernel. Batch output windows to bound memory.
        flat = x.contiguous().reshape(-1, prod(spatial))
        output = torch.empty((flat.shape[0], prod(outspatial)), dtype=x.dtype, device=x.device)
        for linear, coord in enumerate(product(*(range(n) for n in outspatial))):
            locations, included, source_valid = pool_window(op, p, spatial, outspatial, coord)
            ids = torch.tensor(locations, dtype=torch.long, device=x.device)
            is_real = torch.tensor(source_valid, dtype=torch.bool, device=x.device)[None, :]
            mask = torch.tensor(included, dtype=torch.bool, device=x.device)[None, :]
            values = flat[:, ids].masked_fill(~is_real, 0)
            output[:, linear] = _reduce_rows(native, values, mode, mask)
        return output.reshape(output_shape)

    return _candidate(run, hashes,
                      "Gather exact N/C spatial windows, including dilation, ceil bounds or adaptive floor/ceil bins; mask padding indices; select native max or uniform mean. No pooling/math fallback.",
                      op=op, params=p, statistic=mode)


def mean_pooling(case):
    op, p, native, hashes = _prepare(case)

    def run(x, offsets=None, indices=None):
        import torch
        chunk = p["chunk_size"]
        if offsets is None:
            slices = [(start, min(start + chunk, x.shape[1])) for start in range(0, x.shape[1], chunk)]
        else:
            bounds = offsets.tolist()
            slices = [(bounds[s] + c * chunk, min(bounds[s] + (c + 1) * chunk, bounds[s + 1])) for s, c in indices.tolist()]
        outputs = []
        for lo, hi in slices:
            rows = x[:, lo:hi].permute(0, 2, 3, 1).contiguous().reshape(-1, hi - lo)
            outputs.append(_reduce_rows(native, rows, "mean").reshape(x.shape[0], *x.shape[2:]))
        return torch.stack(outputs, dim=1) if outputs else x.new_empty((x.shape[0], 0, *x.shape[2:]))

    return _candidate(run, hashes,
                      "Resolve chunk boundaries each call, permute token axis and select native uniform weighted mean; partial chunks use their actual token count.",
                      op=op, params=p, host=True, statistic="mean")


def bmm(case):
    """Select native raw QK statistics, with no K splitting or partial sums."""
    op, p, native, hashes = _prepare(case)

    def run(a, b):
        import torch
        import torch.nn.functional as F
        batch, m, k = a.shape
        n = b.shape[-1]
        output = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
        for batch_id in range(batch):
            kv = F.pad(b[batch_id].t().contiguous(), (0, 512 - k)).reshape(n, 1, 512)
            for row in range(0, m, 64):
                nr = min(64, m - row)
                query = F.pad(a[batch_id, row:row + nr].contiguous(), (0, 512 - k, 0, 64 - nr))
                for column in range(0, n, 64):
                    nc = min(64, n - column)
                    q = query[None, :, :].expand(nc, -1, -1).contiguous()
                    indices = torch.full((nc, 1, 64), -1, dtype=torch.int32, device=a.device)
                    indices[:, 0, 0] = torch.arange(column, column + nc, dtype=torch.int32, device=a.device)
                    _, dot, _ = native(q, kv, indices, 1.0, 512, None, None)
                    output[batch_id, row:row + nr, column:column + nc] = dot[:, :nr].t().to(a.dtype)
        return output

    return _candidate(run, hashes,
                      "BF16 BMM with K<=512: zero-pad complete A rows/B columns; select one B key for each query row; the exposed max over that singleton is exactly the native QK dot product. Crop/transpose and cast once; no K splitting, added arithmetic, kernel or schedule changes.",
                      op=op, params=p, statistic="max_logits.single_key_QK")


def pointwise_max(case):
    op, p, native, hashes = _prepare(case)
    shape = tuple(p["output"][0])

    def run(x, other=None):
        import torch
        x = x.expand(shape).contiguous().reshape(-1)
        y = torch.zeros_like(x) if op == "ReluFwdOp" else other.expand(shape).contiguous().reshape(-1)
        return _reduce_rows(native, torch.stack((x, y), dim=1), "max").reshape(shape)

    return _candidate(run, hashes,
                      "Broadcast and stack two values per row; ReLU supplies a zero padding value. Select the native max statistic using the same basis-query route as Amax; no pointwise torch math.",
                      op=op, params=p, statistic="max_logits")
