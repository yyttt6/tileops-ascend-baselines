"""Independent CPU references for the three extended MLA manifest contracts.

Only final results are copied back to the inputs' device for the shared verifier.
No baseline implementation, device attention primitive, or vendor runtime is an
oracle. References read the tensors passed to each invocation, including runtime
page/sequence metadata; input generation never caches an expected result.
"""

import math

import torch

from workloads.numerics import Custom, Exact, assert_normalized_error, reference_tolerance
from workloads.workload_base import CallWorkload

__all__ = ["MLAKVCacheWriteCall", "MLAPagedCall", "MLAVarlenCall"]


class _MLACall(CallWorkload):
    def arguments(self):
        # These signatures have no constructor tensors. Avoid allocating the
        # multi-GB input/cache pools a second time merely to read scalar params.
        return self.call.arguments({})


class MLAPagedCall(_MLACall):
    """Absorbed MLA over logical pages, with a shared latent key/value per head.

    The manifest's ref_api, FlashInfer BatchMLAPagedAttentionWrapper.run, returns
    base-2 LSE by default (return_lse_base_on_e=False). The packed varlen op below
    instead follows FlashAttention's natural-log LSE. Neither drops that output.
    """

    def gen_inputs(self):
        inputs = list(super().gen_inputs())
        if inputs[4] is not None:
            # A non-unit, positive per-tensor dequantizer exercises both QK and
            # PV scaling. All original shapes, dtypes and metadata are retained.
            inputs[4].fill_(0.75)
        return tuple(inputs)

    def ref_program(self, q, kv_cache, block_table, cache_seqlens, kv_scale=None):
        p = self.call.params
        batch, query_length, heads, width = q.shape
        rank = p["kv_lora_rank"]
        scale = p["sm_scale"] if p["sm_scale"] is not None else width ** -0.5
        query = q.detach().cpu().float()
        cache = kv_cache.detach().cpu()
        table = block_table.detach().cpu().long()
        lengths = cache_seqlens.detach().cpu().tolist()
        dequant = None if kv_scale is None else kv_scale.detach().cpu().float()
        output = torch.empty((batch, query_length, heads, rank), dtype=q.dtype, device="cpu")
        lse = torch.empty((batch, query_length, heads), dtype=torch.float32, device="cpu")
        page_size = cache.shape[1]
        for b, length in enumerate(lengths):
            pages = table[b, : (length + page_size - 1) // page_size]
            # FP8 CPU advanced indexing is not universally implemented. Gather
            # bytes first, preserving E4M3FN payloads, then decode on the CPU.
            if cache.dtype == torch.float8_e4m3fn:
                rows = cache.view(torch.uint8).index_select(0, pages).view(cache.dtype)
            else:
                rows = cache.index_select(0, pages)
            key = rows.reshape(-1, width)[:length].float()
            if dequant is not None:
                key = key * dequant
            value = key[:, :rank]
            # Slice the query axis so even a long multi-token call does not
            # allocate its full Q x K score matrix at once.
            for start in range(0, query_length, 16):
                end = min(start + 16, query_length)
                scores = torch.matmul(query[b, start:end], key.T) * scale
                if p["is_causal"]:
                    q_positions = length - query_length + torch.arange(start, end, device="cpu")
                    k_positions = torch.arange(length, device="cpu")
                    visible = k_positions[None, :] <= q_positions[:, None]
                    scores.masked_fill_(~visible[:, None, :], -math.inf)
                output[b, start:end] = torch.matmul(scores.softmax(-1), value).to(q.dtype)
                lse[b, start:end] = scores.logsumexp(-1) / math.log(2.0)
        return output.to(q.device), lse.to(q.device)

    def verification(self, *inputs):
        return Exact(normalized=1e-3)


class MLAVarlenCall(_MLACall):
    """Packed prefill with per-head NoPE keys, shared PE keys and natural LSE."""

    def ref_program(self, q, k_nope, k_pe, v, cu_seqlens):
        p = self.call.params
        scale = p["sm_scale"] if p["sm_scale"] is not None else q.shape[-1] ** -0.5
        query, nope, pe, value = (t.detach().cpu() for t in (q, k_nope, k_pe, v))
        offsets = cu_seqlens.detach().cpu().tolist()
        total, heads = q.shape[:2]
        # DV is independent of DN + PE, so slicing q is not an output allocator.
        output = torch.empty((total, heads, v.shape[-1]), dtype=q.dtype, device="cpu")
        lse = torch.empty((total, heads), dtype=torch.float32, device="cpu")
        for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
            length = end - begin
            pe_part = pe[begin:end].float()
            for h0 in range(0, heads, 4):
                h1 = min(h0 + 4, heads)
                key = torch.cat(
                    (
                        nope[begin:end, h0:h1].float(),
                        pe_part[:, None, :].expand(-1, h1 - h0, -1),
                    ),
                    dim=-1,
                ).permute(1, 2, 0)
                values = value[begin:end, h0:h1].float().transpose(0, 1)
                for r0 in range(0, length, 128):
                    r1 = min(r0 + 128, length)
                    # Causality excludes the suffix for every row in this slab.
                    # Every visible key still participates in the same softmax.
                    key_end = r1 if p["is_causal"] else length
                    queries = query[begin + r0:begin + r1, h0:h1].float().transpose(0, 1)
                    scores = torch.matmul(queries, key[..., :key_end]) * scale
                    if p["is_causal"]:
                        visible = (
                            torch.arange(key_end, device="cpu")[None, :]
                            <= torch.arange(r0, r1, device="cpu")[:, None]
                        )
                        scores.masked_fill_(~visible[None, :, :], -math.inf)
                    result = torch.matmul(scores.softmax(-1), values[:, :key_end])
                    output[begin + r0:begin + r1, h0:h1] = result.transpose(0, 1).to(q.dtype)
                    lse[begin + r0:begin + r1, h0:h1] = scores.logsumexp(-1).T
        return output.to(q.device), lse.to(q.device)

    def verification(self, *inputs):
        return Exact(normalized=1e-3)


class MLAKVCacheWriteCall(_MLACall):
    """Scatter latent rows and optional rotated PE, preserving the whole cache.

    RoPE uses the cos || sin table and the manifest's neox/interleaved pairing.
    The rotation is evaluated in FP32, stored in the input dtype, then passed
    through the concat-and-cache conversion. This is the unfused RoPE -> cache
    composition of vLLM's pos_encoding_kernels.cu and cache_kernels.cu. E4M3FN
    conversion divides by the per-tensor scale and saturates to its finite range.
    The native fused vLLM kernel additionally narrows the cos/sin table and
    mutates k_pe; those behaviors are not adopted by this manifest reference.
    """

    def gen_inputs(self):
        inputs = list(super().gen_inputs())
        if inputs[4] is not None:
            inputs[4].fill_(0.75)
        return tuple(inputs)

    def ref_program(
        self, kv_c, k_pe, kv_cache, slot_mapping, scale=None, positions=None, cos_sin_cache=None
    ):
        output = kv_cache.detach().cpu().clone()
        flat = output.view(-1, output.shape[-1])
        slots = slot_mapping.detach().cpu().long()
        if ((slots < -1) | (slots >= flat.shape[0])).any().item():
            raise ValueError("MLA cache slots must be -1 or valid physical row indices")
        tokens = (slots >= 0).nonzero().flatten()
        destinations = slots[tokens]
        if destinations.unique().numel() != destinations.numel():
            raise ValueError("MLA cache writes require distinct non-negative slots")
        latent = kv_c.detach().cpu().index_select(0, tokens)
        positional = k_pe.detach().cpu().index_select(0, tokens)
        if self.call.params["fuse_rope"]:
            selected_positions = positions.detach().cpu().long().index_select(0, tokens)
            trig = cos_sin_cache.detach().cpu().index_select(0, selected_positions).float()
            cosine, sine = trig.chunk(2, dim=-1)
            positional = positional.float()
            layout = self.call.params["rope_layout"]
            if layout == "neox":
                first, second = positional.chunk(2, dim=-1)
                positional = torch.cat(
                    (first * cosine - second * sine, second * cosine + first * sine), dim=-1
                )
            elif layout == "interleaved":
                first, second = positional[:, ::2], positional[:, 1::2]
                positional = torch.stack(
                    (first * cosine - second * sine, second * cosine + first * sine), dim=-1
                ).flatten(-2)
            else:
                raise ValueError(f"Unknown MLA RoPE layout: {layout!r}")
            positional = positional.to(k_pe.dtype)
        stored = torch.cat((latent, positional), dim=-1)
        if output.dtype == torch.float8_e4m3fn:
            limit = torch.finfo(output.dtype).max
            stored = (stored.float() / scale.detach().cpu().float()).clamp(-limit, limit)
        stored = stored.to(output.dtype)
        # Byte scatter works for FP8 on CPU and preserves every untouched bit,
        # including signed zeros and NaN payloads in the original cache pool.
        flat.view(torch.uint8).index_copy_(0, destinations, stored.contiguous().view(torch.uint8))
        return output.to(kv_cache.device)

    def verification(self, *inputs):
        slots = inputs[3].detach().cpu().long()
        width = inputs[2].shape[-1]
        latent_width = inputs[0].shape[-1]
        fuse_rope = self.call.params["fuse_rope"]

        def validate(got, expected):
            actual = got.detach().cpu().reshape(-1, width)
            target = expected.detach().cpu().reshape(-1, width)
            written = torch.zeros(target.shape[0], dtype=torch.bool, device="cpu")
            written[slots[slots >= 0]] = True
            for begin in range(0, target.shape[0], 1024):
                end = min(begin + 1024, target.shape[0])
                a, b = actual[begin:end], target[begin:end]
                if not fuse_rope or target.dtype == torch.float8_e4m3fn:
                    # Copy/quantization outputs require exactly the stored bytes.
                    torch.testing.assert_close(
                        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8),
                        atol=0, rtol=0,
                    )
                    continue
                selected = written[begin:end]
                torch.testing.assert_close(
                    a[~selected].contiguous().view(torch.uint8),
                    b[~selected].contiguous().view(torch.uint8), atol=0, rtol=0,
                )
                torch.testing.assert_close(
                    a[selected, :latent_width].contiguous().view(torch.uint8),
                    b[selected, :latent_width].contiguous().view(torch.uint8), atol=0, rtol=0,
                )
                if selected.any().item():
                    rotated = a[selected, latent_width:]
                    expected_rotated = b[selected, latent_width:]
                    torch.testing.assert_close(
                        rotated, expected_rotated, **reference_tolerance(target.dtype)
                    )
                    assert_normalized_error(rotated, expected_rotated)

        return Custom(validate, "Entire MLA cache: exact untouched/latent bytes and complete rotated PE")
