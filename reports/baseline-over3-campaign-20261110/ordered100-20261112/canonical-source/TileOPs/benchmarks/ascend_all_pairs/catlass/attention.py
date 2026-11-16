"""Dense, packed and paged layouts of original CATLASS example49 attention."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
from .build import build, hashes


def _check(q, k, v, params, extras=()):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    if (q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype
            or q.shape[-1] > 128 or k.shape[-1] != q.shape[-1] or k.shape != v.shape):
        raise UnsupportedCase("Original example49 has matching FP16/BF16 Q/K/V and fixed D<=128 PV output tile.")
    if params.get("softcap") not in (None, 0, 0.0):
        raise UnsupportedCase("Original example49 does not apply score softcapping.")
    if params.get("pos_encoding_mode", "none") != "none" or any(x is not None for x in extras):
        raise UnsupportedCase("Original example49 has no RoPE, scaled-FP8 or attention-sink input.")
    if params.get("out_dtype") not in (None, str(q.dtype).removeprefix("torch.")):
        raise UnsupportedCase("Original example49 writes the operand dtype.")
    if q.shape[-2] % k.shape[-2]:
        raise UnsupportedCase("Q heads must be a multiple of KV heads")


class AttentionEngine:
    def __init__(self, params):
        from benchmarks.ascend_baselines import _owned_native
        self.params = dict(params)
        binary, self.source_sha256, self.receipt = build("attention_cases")
        self.lib = _owned_native(binary, ["classic_catlass_fa_cases_tiling_size",
                                         "classic_catlass_fa_cases_prepare", "classic_catlass_fa_cases_run"])
        self.lib.classic_catlass_fa_cases_tiling_size.restype = C.c_uint64
        self.lib.classic_catlass_fa_cases_prepare.argtypes = [C.c_void_p, C.POINTER(C.c_int64), C.c_float]
        self.lib.classic_catlass_fa_cases_prepare.restype = C.c_int
        self.lib.classic_catlass_fa_cases_run.argtypes = [C.c_void_p] * 9 + [C.c_int]
        self.lib.classic_catlass_fa_cases_run.restype = C.c_int

    def _launch(self, q, k, v, causal):
        import torch
        batch, sq, hq, dim = q.shape
        original_dim = dim
        # The original PV/output path uses its fixed 128-wide template even
        # when the host embed is smaller. Zero-pad heads to that public shape,
        # preserve the original softmax scale, then crop the output channels.
        if dim < 128:
            import torch.nn.functional as F
            q, k, v = (F.pad(value, (0, 128 - dim)) for value in (q, k, v))
            dim = 128
        sk, hk = k.shape[1:3]
        if min(batch, sq, sk, hq, hk, dim) <= 0:
            raise ValueError("Native attention requires nonempty extents")
        masked = causal and sq > 1
        # Tiling depends on current sequence metadata; rebuild it inside run.
        host = C.create_string_buffer(self.lib.classic_catlass_fa_cases_tiling_size())
        sizes = (C.c_int64 * 7)(batch, sq, sk, hq, hk, dim, 2 if masked else 0)
        scale = self.params.get("sm_scale")
        scale = original_dim ** -0.5 if scale is None else float(scale)
        if self.lib.classic_catlass_fa_cases_prepare(host, sizes, scale):
            raise RuntimeError("Original CATLASS example49 tiling rejected the dimensions")
        tiling = torch.tensor(list(host.raw), dtype=torch.uint8, device=q.device)
        q_lengths = torch.full((batch,), sq, dtype=torch.int64, device=q.device)
        k_lengths = torch.full((batch,), sk, dtype=torch.int64, device=q.device)
        mask = None
        if masked:
            qp = torch.arange(sq, device=q.device) + sk - sq
            kp = torch.arange(sk, device=q.device)
            mask = (kp[None, :] > qp[:, None]).to(torch.uint8).unsqueeze(0).expand(batch, sq, sk).contiguous()
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        storage = torch.empty(2 * q.numel(), dtype=q.dtype, device=q.device)
        rc = self.lib.classic_catlass_fa_cases_run(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), storage.data_ptr(),
            q_lengths.data_ptr(), k_lengths.data_ptr(), tiling.data_ptr(),
            None if mask is None else mask.data_ptr(), torch.npu.current_stream(q.device).npu_stream,
            0 if q.dtype == torch.float16 else 1)
        if rc:
            raise RuntimeError(f"Original CATLASS example49 launch status {rc}")
        return storage[:q.numel()].view(q.shape)[..., :original_dim].contiguous()

    def __call__(self, q, k, v):
        import torch
        sq, sk = q.shape[1], k.shape[1]
        p = self.params
        causal = bool(p.get("is_causal", False))
        left, right = p.get("window_size_left", -1), p.get("window_size_right", -1)
        if sq <= sk and left == -1 and right == -1:
            return self._launch(q, k, v, causal)
        # Crop each independent query's visible contiguous KV interval. This
        # changes neither the softmax reduction nor the underlying kernel.
        out = torch.zeros_like(q)
        for i in range(sq):
            pos = sk - sq + i
            lo = max(0, pos - left) if left >= 0 else 0
            hi = min(sk, pos + right + 1) if right >= 0 else sk
            if causal:
                hi = min(hi, pos + 1)
            if hi > lo:
                out[:, i:i + 1].copy_(self._launch(q[:, i:i + 1], k[:, lo:hi], v[:, lo:hi], False))
        return out


def _candidate(run, engine, description):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    return Candidate(Implementation(run), "CATLASS example49 FAInferTla FP16/BF16 original kernel and host tiling",
                     description, engine.source_sha256 | hashes([Path(__file__)]),
                     timing_mode="host_roundtrip",
                     limitations=("No current-host numerical receipt; metadata host reads and every packing/launch count in timing.",))


def dense(case):
    q, k, v, *extras = case.inputs
    _check(q, k, v, case.params, extras)
    engine = AttentionEngine(case.params)
    def run(q, k, v, *extras):
        _check(q, k, v, case.params, extras)
        return engine(q, k, v)
    return _candidate(run, engine, "BSND packing; original causal mask, or per-query KV interval cropping for windows. Empty visibility uses the manifest zero identity; no softmax decomposition.")


def _bounds(metadata, total):
    bounds = metadata.tolist()
    if not bounds or bounds[0] != 0 or bounds[-1] != total or any(a > b for a, b in zip(bounds, bounds[1:])):
        raise ValueError("Invalid current packed sequence bounds")
    return bounds


def varlen(case):
    q, k, v, cq, ck, *extras = case.inputs
    _check(q, k, v, case.params, extras)
    engine = AttentionEngine(case.params)
    def run(q, k, v, cq, ck, *extras):
        import torch
        _check(q, k, v, case.params, extras)
        qs, ks = _bounds(cq, q.shape[0]), _bounds(ck, k.shape[0])
        if len(qs) != len(ks):
            raise ValueError("Q and KV batch counts disagree")
        out = torch.zeros_like(q)
        for qa, qb, ka, kb in zip(qs, qs[1:], ks, ks[1:]):
            if qb > qa and kb > ka:
                out[qa:qb].copy_(engine(q[qa:qb][None], k[ka:kb][None], v[ka:kb][None])[0])
        return out
    return _candidate(run, engine, "Read cu_seqlens each run; split independent packed sequences, native full attention per sequence; restore packed order. No cached input metadata.")


def paged(case):
    q, k, v, table, lengths, cq, *extras = case.inputs
    _check(q, k, v, case.params, extras)
    engine = AttentionEngine(case.params)
    def run(q, k, v, table, lengths, cq, *extras):
        import torch
        _check(q, k, v, case.params, extras)
        qs, lens = _bounds(cq, q.shape[0]), lengths.tolist()
        if len(lens) != len(qs) - 1:
            raise ValueError("Paged attention batch counts disagree")
        out = torch.zeros_like(q)
        page_size = k.shape[1]
        for b, (qa, qb, length) in enumerate(zip(qs, qs[1:], lens)):
            pages = (length + page_size - 1) // page_size
            if length < 0 or pages > table.shape[1]:
                raise ValueError("Cache length outside page table capacity")
            if qb > qa and length:
                indices = table[b, :pages].long()
                keys = k[indices].flatten(0, 1)[:length]
                values = v[indices].flatten(0, 1)[:length]
                out[qa:qb].copy_(engine(q[qa:qb][None], keys[None], values[None])[0])
        return out
    return _candidate(run, engine, "Gather current fragmented pages in logical order and crop final page on every run; independent native complete attention per request; preserve packed queries and caches.")


def mha_paged(case):
    q, k, v, lengths, table = case.inputs
    _check(q, k, v, case.params)
    engine = AttentionEngine(case.params)
    page_size = case.params["page_size"]
    def run(q, k, v, lengths, table):
        import torch
        keys, values = k.reshape(-1, page_size, *k.shape[1:]), v.reshape(-1, page_size, *v.shape[1:])
        lens = lengths.tolist()
        out = torch.zeros_like(q)
        if len(lens) != q.shape[0]:
            raise ValueError("MHA cache length count differs from batch")
        for b, length in enumerate(lens):
            pages = (length + page_size - 1) // page_size
            if length < 0 or pages > table.shape[1]:
                raise ValueError("MHA cache length outside page table capacity")
            if length:
                indices = table[b, :pages].long()
                kb = keys[indices].flatten(0, 1)[:length]
                vb = values[indices].flatten(0, 1)[:length]
                out[b:b + 1].copy_(engine(q[b:b + 1], kb[None], vb[None]))
        return out
    return _candidate(run, engine, "View flat physical KV storage as pages; gather current block table and crop current cache lengths each run; original native attention per batch item.")


def prefill_append(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    q, kn, vn, kp, vp, *_ = case.inputs
    p = case.params
    _check(q, kn, vn, p)
    if p.get("fuse_rope") or kp.dtype != q.dtype or vp.dtype != q.dtype:
        raise UnsupportedCase("Native attention append adapter requires unquantized caches and no fused RoPE")
    engine = AttentionEngine(p)
    page_size = p["page_size"]
    def run(q, kn, vn, kp, vp, k_scale, v_scale, cq, lengths, table):
        import torch
        qs, lens = _bounds(cq, q.shape[0]), lengths.tolist()
        out = torch.empty_like(q)
        if len(lens) != len(qs) - 1 or table.shape[0] != len(lens):
            raise ValueError("Append batch metadata counts disagree")
        keys, values = kp.clone(), vp.clone()
        appends = []
        for b, (qa, qb, old_len) in enumerate(zip(qs, qs[1:], lens)):
            if old_len < 0 or old_len + qb - qa > table.shape[1] * page_size:
                raise ValueError("Append exceeds current cache table capacity")
            pos = torch.arange(old_len, device=q.device)
            rows = table[b, pos // page_size].long() * page_size + pos % page_size
            k_all = torch.cat((kp[rows], kn[qa:qb]))
            v_all = torch.cat((vp[rows], vn[qa:qb]))
            if qb > qa:
                out[qa:qb].copy_(engine(q[qa:qb][None], k_all[None], v_all[None])[0])
            new_pos = torch.arange(old_len, old_len + qb - qa, device=q.device)
            new_rows = table[b, new_pos // page_size].long() * page_size + new_pos % page_size
            appends.append((new_rows, qa, qb))
        # Read all requests from the original caches before applying writes,
        # preserving the complete oracle's behavior for shared physical pages.
        for rows, qa, qb in appends:
            keys[rows], values[rows] = kn[qa:qb], vn[qa:qb]
        return out, keys, values
    return _candidate(run, engine, "Read current packed bounds/cache lengths/table; gather original cache and concatenate new rows for native attention; copy-on-write append returns complete output, K cache and V cache; original input caches protected.")


def nsa_selected(case):
    q, k, v, *_ = case.inputs
    p = case.params
    _check(q, k, v, p)
    # Selected positions already encode visibility; native attention sees the
    # complete selected multiset and must not apply an additional causal mask.
    params = p | {"sm_scale": p.get("scale"), "is_causal": False}
    engine = AttentionEngine(params)
    block_size = p["block_size"]
    def run(q, k, v, block_indices, block_counts, offsets, token_indices):
        import torch
        boundaries = _bounds(offsets, q.shape[0])
        selected, counts = block_indices.tolist(), block_counts.tolist()
        hkv = k.shape[1]
        group = q.shape[1] // hkv
        out = torch.zeros_like(q)
        for start, end in zip(boundaries, boundaries[1:]):
            for token in range(start, end):
                limit = token - start + 1 if p["is_causal"] else end - start
                for head in range(hkv):
                    count = counts[token][head]
                    if not 0 <= count <= len(selected[token][head]):
                        raise ValueError("Selected block count outside index slots")
                    positions = [start + block * block_size + offset
                                 for block in selected[token][head][:count]
                                 for offset in range(block_size)
                                 if 0 <= block * block_size + offset < limit]
                    if positions:
                        # Keep repeated block indices: attention is over slots,
                        # not over a deduplicated set of physical tokens.
                        indices = torch.tensor(positions, dtype=torch.int64, device=q.device)
                        kb, vb = k[indices, head:head + 1], v[indices, head:head + 1]
                        first, last = head * group, (head + 1) * group
                        value = engine(q[token:token + 1, first:last][None], kb[None], vb[None])
                        out[token:token + 1, first:last].copy_(value[0])
        return out
    return _candidate(run, engine, "Current selected-block/count metadata to exact slot multiset, retaining duplicates and causal bounds; gather KV per token/head then invoke original complete attention with original scale; zero empty visibility only. No sparse score or softmax emulation.")
