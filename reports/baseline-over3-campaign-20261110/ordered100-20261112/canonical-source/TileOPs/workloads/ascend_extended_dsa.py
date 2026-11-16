"""Independent CPU workload for the manifest's DeepSeek V3.2 paged DSA.

This is the 576-wide, 656-byte FP8 cache contract, not the newer V4.1 format.
Byte layout and BF16 dequantization are documented in FlashMLA's
``tests/quant.py`` at commit 082094b793fcc7452977d0a71a00e266a2e3061e;
``tests/ref.py`` at that commit establishes natural-log decode LSE and the
zero-output / +inf-LSE convention for a query with no selected keys. These
sources are documentation only: no provider implementation is imported.

The manifest takes logical positions and returns LSE in [B, S_q, H] order.
FlashMLA's native API instead takes physical slots and returns [B, H, S_q].
"""

import math

import torch

from workloads.numerics import Exact
from workloads.workload_base import CallWorkload

__all__ = ["DSAPagedCall"]

_LATENT = 512
_ROPE = 64
_GROUP = 128
_GROUPS = _LATENT // _GROUP
_SCALE_END = _LATENT + 4 * _GROUPS
_ROW_BYTES = _SCALE_END + 2 * _ROPE
_ROWS_PER_CHUNK = 4096


def _make_cache(shape, generator):
    """Build every byte of every cache row on CPU, with bounded temporaries.

    The generated positive FP32 scales are per token and per 128-value group.
    Like the V3.2 source quantizer, use a power-of-two scale covering amax/448.
    Distinct group gains exercise the four scale locations. No unquantized
    cache or expected attention result is retained by the workload.
    """
    cache = torch.empty(shape, dtype=torch.uint8, device="cpu")
    rows = cache.view(-1, _ROW_BYTES)
    gain = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32, device="cpu")
    for begin in range(0, rows.shape[0], _ROWS_PER_CHUNK):
        end = min(begin + _ROWS_PER_CHUNK, rows.shape[0])
        count = end - begin
        latent = torch.randn(
            (count, _GROUPS, _GROUP), dtype=torch.float32,
            device="cpu", generator=generator,
        )
        latent = (latent * gain[None, :, None]).to(torch.bfloat16).float()
        scales = (latent.abs().amax(-1) / 448.0).clamp_min(1e-4)
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
        codes = (latent / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
        rope = torch.randn(
            (count, _ROPE), dtype=torch.float32, device="cpu", generator=generator,
        ).to(torch.bfloat16)
        rows[begin:end, :_LATENT] = codes.view(torch.uint8).reshape(count, _LATENT)
        rows[begin:end, _LATENT:_SCALE_END] = scales.contiguous().view(torch.uint8)
        rows[begin:end, _SCALE_END:] = rope.contiguous().view(torch.uint8)
    return cache


def _decode_rows(rows):
    """Decode selected CPU uint8 rows; preserve the V3.2 BF16 storage rounding."""
    # Make fields contiguous before reinterpreting them as wider element types.
    # In particular, the 16 scale bytes are floats, not four numeric uint8 codes.
    codes = rows[:, :_LATENT].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows[:, _LATENT:_SCALE_END].contiguous().view(torch.float32)
    latent = (codes.reshape(-1, _GROUPS, _GROUP) * scales.unsqueeze(-1))
    latent = latent.reshape(-1, _LATENT).to(torch.bfloat16)
    rope = rows[:, _SCALE_END:].contiguous().view(torch.bfloat16)
    return torch.cat((latent, rope), dim=-1).double()


class DSAPagedCall(CallWorkload):
    """Paged sparse MLA with complete output and LSE, leaving all inputs intact."""

    def arguments(self):
        # This op has only sm_scale and no constructor tensors. The base method
        # would unnecessarily materialize the 1.28 GiB cache a second time.
        return self.call.arguments({})

    def gen_inputs(self):
        """Keep manifest metadata verbatim; generate a valid, fully populated cache."""
        q_spec = self.call.specs["q"]
        cache_spec = self.call.specs["kv_cache"]
        q = torch.randn(
            q_spec.shape, dtype=torch.float32, device="cpu", generator=self.rng("q"),
        ).to(torch.bfloat16)
        cache = _make_cache(cache_spec.shape, self.rng("kv_cache"))
        values = {"q": q, "kv_cache": cache}
        for name in ("block_table", "cache_seqlens", "indices"):
            spec = self.call.specs[name]
            if spec.values is None:
                raise ValueError(f"DSAPagedFwdOp needs manifest-generated {name}")
            values[name] = torch.tensor(
                spec.values, dtype=getattr(torch, spec.dtype), device="cpu",
            ).reshape(spec.shape)
        device = self.device
        return tuple(values[name].to(device) for name in self.call.signature.inputs)

    def ref_program(self, q, kv_cache, block_table, cache_seqlens, indices):
        """Evaluate only the supplied selected rows, without caching input-dependent data.

        All decoding, indexing and attention arithmetic run on CPU. Duplicated
        indices contribute once per occurrence. Only -1 is padding; other
        out-of-range indices violate the manifest rather than silently masking.
        The selection is the entire mask: do not impose a second causal mask.
        """
        if q.ndim != 4 or q.shape[-1] != _LATENT + _ROPE or q.dtype != torch.bfloat16:
            raise ValueError("q must be bfloat16 [B, S_q, H, 576]")
        if (kv_cache.ndim != 3 or kv_cache.shape[-1] != _ROW_BYTES
                or kv_cache.dtype != torch.uint8 or not kv_cache.is_contiguous()):
            raise ValueError("kv_cache must be contiguous uint8 [NP, PS, 656]")
        batch, queries, heads, _ = q.shape
        pages, page_size, _ = kv_cache.shape
        if (block_table.ndim != 2 or block_table.shape[0] != batch
                or block_table.dtype != torch.int32):
            raise ValueError("block_table must be int32 [B, W]")
        if cache_seqlens.shape != (batch,) or cache_seqlens.dtype != torch.int32:
            raise ValueError("cache_seqlens must be int32 [B]")
        if (indices.ndim != 3 or indices.shape[:2] != (batch, queries)
                or indices.dtype != torch.int32):
            raise ValueError("indices must be int32 [B, S_q, K]")
        if queries <= 0 or page_size <= 0:
            raise ValueError("S_q and PS must be positive")
        scale = self.call.params["sm_scale"]
        scale = (_LATENT + _ROPE) ** -0.5 if scale is None else float(scale)
        if not math.isfinite(scale):
            raise ValueError("sm_scale must be finite")

        query = q.detach().cpu().double()
        cache = kv_cache.detach().cpu()
        table = block_table.detach().cpu().long()
        lengths = cache_seqlens.detach().cpu().long()
        selected = indices.detach().cpu().long()
        if bool(((table < 0) | (table >= pages)).any()):
            raise ValueError("block_table entries must lie in [0, NP)")
        if bool(((lengths < queries) | (lengths > table.shape[1] * page_size)).any()):
            raise ValueError("cache_seqlens entries must lie in [S_q, W * PS]")
        invalid = (selected != -1) & ((selected < 0) | (selected >= lengths[:, None, None]))
        if bool(invalid.any()):
            raise ValueError("each index must be -1 or a position below cache_seqlens[b]")

        output = torch.zeros(
            (batch, queries, heads, _LATENT), dtype=torch.bfloat16, device="cpu",
        )
        lse = torch.full(
            (batch, queries, heads), float("inf"), dtype=torch.float32, device="cpu",
        )
        for b in range(batch):
            for s in range(queries):
                positions = selected[b, s]
                positions = positions[positions != -1]
                if positions.numel() == 0:
                    # FlashMLA decode's explicit empty-selection convention.
                    continue
                physical_pages = table[b, positions // page_size]
                rows = cache[physical_pages, positions % page_size]
                key = _decode_rows(rows)
                scores = (query[b, s] @ key.T) * scale
                row_lse = torch.logsumexp(scores, dim=-1)
                probabilities = torch.exp(scores - row_lse.unsqueeze(-1))
                output[b, s] = (probabilities @ key[:, :_LATENT]).to(torch.bfloat16)
                lse[b, s] = row_lse.to(torch.float32)
        # Transfers only; all numerical reference work above is performed on CPU.
        return output.to(q.device), lse.to(q.device)

    def verification(self, *inputs):
        # Check BOTH declared outputs with the unchanged shared dtype tolerances.
        return Exact()
