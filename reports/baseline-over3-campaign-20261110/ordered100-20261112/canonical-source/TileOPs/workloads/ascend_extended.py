"""Independent references for manifest entries missing a benchmark factory.

These workloads implement the manifest contract, including declared mutations.
References run on CPU so a custom Ascend OPP cannot intercept its own oracle.
No baseline provider or vendor kernel is used by a reference.
"""

import torch
import torch.nn.functional as F

from workloads.numerics import Custom, Exact, assert_normalized_error, reference_tolerance
from workloads.workload_base import CallWorkload


def _causal_verification():
    def validate(got, expected):
        assert isinstance(got, (tuple, list)) and len(got) == 2
        torch.testing.assert_close(got[0], expected[0], **reference_tolerance(expected[0].dtype))
        assert_normalized_error(got[0], expected[0])
        # Updating the history only copies/reorders existing values.
        torch.testing.assert_close(got[1], expected[1], atol=0, rtol=0)

    return Custom(validate, "Complete convolution output plus exact copied history state")


class CausalConvCall(CallWorkload):
    """Packed depthwise causal convolution and the complete updated state."""

    def ref_program(self, x, weight, conv_state, cu_seqlens, bias=None, has_initial_state=None):
        device, dtype = x.device, x.dtype
        xx, ww, state = x.cpu().float(), weight.cpu().float(), conv_state.cpu().clone()
        offsets = cu_seqlens.cpu().tolist()
        bb = None if bias is None else bias.cpu().float()
        flags = None if has_initial_state is None else has_initial_state.cpu().tolist()
        width = weight.shape[-1]
        result = torch.empty_like(xx)
        for i, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            history = state[i].float() if flags is not None and flags[i] else torch.zeros_like(state[i]).float()
            tokens = torch.cat((history, xx[:, begin:end]), dim=-1)
            y = F.conv1d(tokens.unsqueeze(0), ww.unsqueeze(1), bias=bb, groups=x.shape[0])[0]
            if self.call.params["activation"] == "silu":
                y = F.silu(y)
            result[:, begin:end] = y
            state[i].copy_(tokens[:, -(width - 1):])
        return result.to(device=device, dtype=dtype), state.to(device)

    def verification(self, *inputs):
        return _causal_verification()


class CausalConvUpdateCall(CallWorkload):
    """One decode token per sequence, returning output and shifted state."""

    def ref_program(self, x, weight, conv_state, bias=None):
        state = torch.cat((conv_state.cpu()[..., 1:], x.cpu().unsqueeze(-1)), dim=-1)
        result = (state.float() * weight.cpu().float().unsqueeze(0)).sum(-1)
        if bias is not None:
            result += bias.cpu().float()
        if self.call.params["activation"] == "silu":
            result = F.silu(result)
        return result.to(device=x.device, dtype=x.dtype), state.to(x.device)

    def verification(self, *inputs):
        return _causal_verification()


class PagedCacheGatherCall(CallWorkload):
    """Gather the full declared destination in logical sequence order."""

    def gen_inputs(self):
        values = list(super().gen_inputs())
        if values[5] is not None:
            values[5].fill_(0.75)
        return tuple(values)

    def ref_program(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None):
        source = cache.cpu()
        if source.dtype == torch.float8_e4m3fn:
            source = source.float()
        table = block_table.cpu().long()
        offsets = cu_seq_lens.cpu().tolist()
        starts = [0] * (len(offsets) - 1) if seq_starts is None else seq_starts.cpu().tolist()
        output = torch.empty(dst.shape, dtype=dst.dtype, device="cpu")
        page_size = source.shape[1]
        for batch, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            positions = torch.arange(starts[batch], starts[batch] + end - begin)
            page_ids = table[batch, positions // page_size]
            values = source[page_ids, positions % page_size]
            if scale is not None:
                values = values.float() * scale.cpu().float()
            output[begin:end] = values.to(dst.dtype)
        return output.to(dst.device)


class Int8GemmCall(CallWorkload):
    """INT32 dot products and independent scale/bias evaluation."""

    def gen_inputs(self):
        values = list(super().gen_inputs())
        # Nonunit positive quantizer scales exercise both broadcast directions.
        for i in (2, 3):
            values[i].copy_(values[i].abs() * 0.01 + 0.005)
        return tuple(values)

    def ref_program(self, a, b, scale_a, scale_b, bias=None):
        # The manifest K fits an exact signed INT32 accumulator, even at int8 extrema.
        result = (a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).T).double()
        result *= scale_a.cpu().double()
        result *= scale_b.cpu().double()
        if bias is not None:
            result += bias.cpu().double()
        return result.to(device=a.device, dtype=getattr(torch, self.call.params["out_dtype"]))

    def verification(self, *inputs):
        return Exact(normalized=1e-3)


class MergeAttentionCall(CallWorkload):
    """Merge natural-log states using the manifest's explicit exp/log formula.

    FlashInfer's native merge_state consumes base-2 states. A provider using
    that API must convert the log units in both directions. Two empty states
    produce zero values and -inf LSE, as the manifest explicitly requires.
    """

    def ref_program(self, v_a, s_a, v_b, s_b):
        a, b = s_a.cpu().double(), s_b.cpu().double()
        maximum = torch.maximum(a, b)
        empty = torch.isneginf(maximum)
        safe_maximum = torch.where(empty, torch.zeros_like(maximum), maximum)
        wa, wb = torch.exp(a - safe_maximum), torch.exp(b - safe_maximum)
        denominator = wa + wb
        va = torch.where((wa != 0).unsqueeze(-1), v_a.cpu().double(), 0.0)
        vb = torch.where((wb != 0).unsqueeze(-1), v_b.cpu().double(), 0.0)
        result = (va * wa.unsqueeze(-1) + vb * wb.unsqueeze(-1)) / torch.where(
            empty, torch.ones_like(denominator), denominator
        ).unsqueeze(-1)
        lse = safe_maximum + torch.log(denominator)
        return result.to(device=v_a.device, dtype=v_a.dtype), lse.to(device=s_a.device, dtype=s_a.dtype)

    def verification(self, *inputs):
        return Exact(normalized=1e-3)


class SelectiveScanCall(CallWorkload):
    """Independent CPU S6 recurrence, including packed resets and final state.

    The recurrence follows state-spaces/mamba's selective_scan_ref; the manifest
    additionally specifies an initial state and reset on a change of sequence ID.
    Group broadcasting avoids materializing [B, D, N, S] intermediates.
    """

    def gen_inputs(self):
        values = list(super().gen_inputs())
        # Stable decays exercise the full long sequence without exponential overflow.
        values[2].copy_(-values[2].abs() - 0.01)
        for index in (0, 3, 4):
            values[index].mul_(0.1)
        if values[8] is not None:
            seq = torch.arange(values[0].shape[-1], device=values[8].device)
            values[8].copy_((seq >= values[0].shape[-1] // 2).to(torch.int32))
        return tuple(values)

    def ref_program(self, u, delta, A, B_in, C_in, D_in=None, z=None,
                    delta_bias=None, seq_idx=None, initial_states=None):
        batch, channels, length = u.shape
        groups, states = B_in.shape[1:3]
        heads = channels // groups
        source = u.cpu().double().reshape(batch, groups, heads, length)
        step = delta.cpu().double().reshape_as(source)
        if delta_bias is not None:
            step = step + delta_bias.cpu().double().reshape(1, groups, heads, 1)
        if self.call.params["delta_softplus"]:
            step = F.softplus(step)
        decay = A.cpu().double().reshape(1, groups, heads, states)
        b, c = B_in.cpu().double(), C_in.cpu().double()
        state = (torch.zeros(batch, groups, heads, states, dtype=torch.float64)
                 if initial_states is None else
                 initial_states.cpu().double().reshape(batch, groups, heads, states).clone())
        sequence = None if seq_idx is None else seq_idx.cpu()
        output = torch.empty(batch, groups, heads, length, dtype=torch.float64)
        for token in range(length):
            if sequence is not None and token:
                reset = sequence[:, token] != sequence[:, token - 1]
                state[reset] = 0
            dt = step[..., token, None]
            state = (torch.exp(dt * decay) * state
                     + dt * b[:, :, None, :, token] * source[..., token, None])
            output[..., token] = (state * c[:, :, None, :, token]).sum(-1)
        output = output.reshape(batch, channels, length)
        if D_in is not None:
            output += u.cpu().double() * D_in.cpu().double().reshape(1, channels, 1)
        if z is not None:
            output *= F.silu(z.cpu().double())
        return (output.to(device=u.device, dtype=u.dtype),
                state.reshape(batch, channels, states).to(device=u.device, dtype=torch.float32))

    def verification(self, *inputs):
        return Exact(normalized=1e-3)


class FusedQKNormRoPECall(CallWorkload):
    """Return the entire mutated QKV, including the unchanged V columns."""

    def ref_program(self, qkv, q_weight, k_weight, cos_sin_cache, positions):
        p = self.call.params
        width = q_weight.numel()
        heads, kv_heads = p["num_heads"], p["num_kv_heads"]
        output = qkv.cpu().clone().reshape(qkv.shape[0], heads + 2 * kv_heads, width)
        cache = cos_sin_cache.cpu().float()[positions.cpu().long()]
        rotary = cache.shape[-1]
        cos, sin = (part[:, None, :] for part in cache.chunk(2, dim=-1))
        for start, count, weight in ((0, heads, q_weight), (heads, kv_heads, k_weight)):
            value = output[:, start:start + count].float()
            # The named fused ref_api retains FP32 normalized values through RoPE.
            value = value * (torch.rsqrt(value.square().mean(-1, keepdim=True) + p["eps"])
                             * weight.cpu().float())
            rotated = value[..., :rotary]
            if p["rope_layout"] == "neox":
                first, second = rotated.chunk(2, dim=-1)
                rotated = torch.cat((first * cos - second * sin,
                                     second * cos + first * sin), dim=-1)
            else:
                first, second = rotated[..., ::2], rotated[..., 1::2]
                rotated = torch.stack((first * cos - second * sin,
                                       second * cos + first * sin), dim=-1).flatten(-2)
            value[..., :rotary] = rotated
            output[:, start:start + count] = value.to(qkv.dtype)
        return output.reshape_as(qkv).to(qkv.device)

    def verification(self, *inputs):
        p = self.call.params
        qk_end = (p["num_heads"] + p["num_kv_heads"]) * inputs[1].numel()

        def validate(got, expected):
            torch.testing.assert_close(got, expected, **reference_tolerance(expected.dtype))
            assert_normalized_error(got, expected)
            torch.testing.assert_close(got[..., qk_end:], expected[..., qk_end:], atol=0, rtol=0)

        return Custom(validate, "Complete normalized/rotated QK and exact unchanged V")


class FusedAddRMSNormQuantCall(CallWorkload):
    """Residual RMSNorm and dynamic quantization under the manifest contract.

    The ref_api establishes FP32 residual accumulation and the two input-dtype
    roundings in normalization. The manifest explicitly caps the *scale*, whereas
    upstream vLLM caps amax. Its written scale formula also has no positive floor;
    an all-zero row consequently returns scale zero and quantized zeros.
    """

    def gen_inputs(self):
        x, residual, weight = super().gen_inputs()
        # Keep nonzero rows and expose the declared zero-scale convention.
        if x.numel() > x.shape[-1]:
            x.reshape(-1, x.shape[-1])[0].zero_()
            residual.reshape(-1, x.shape[-1])[0].zero_()
        cap = self.call.params["scale_ub"]
        if cap is not None:
            dtype = getattr(torch, self.call.params["quant_dtype"])
            maximum = 127.0 if dtype == torch.int8 else torch.finfo(dtype).max
            # The clamped case must actually reach its scale cap. Its BF16
            # weight domain safely represents this finite, varying magnitude.
            weight.copy_((weight.float().abs() + 1) * (maximum * cap * 2))
        return x, residual, weight

    def ref_program(self, x, residual, weight):
        p = self.call.params
        value = x.cpu().float() + residual.cpu().float()
        residual_out = value.to(x.dtype)
        inv_rms = torch.rsqrt(value.square().mean(-1, keepdim=True) + p["eps"])
        normalized = ((value * inv_rms).to(x.dtype) * weight.cpu()).float()
        dtype = getattr(torch, p["quant_dtype"])
        maximum = 127.0 if dtype == torch.int8 else torch.finfo(dtype).max
        scale = normalized.abs().amax(-1, keepdim=True) / maximum
        if p["scale_ub"] is not None:
            scale = scale.clamp(max=p["scale_ub"])
        divided = normalized / torch.where(scale == 0, torch.ones_like(scale), scale)
        if dtype == torch.int8:
            divided = divided.round()
        minimum = -128.0 if dtype == torch.int8 else -maximum
        quantized = divided.clamp(minimum, maximum).to(dtype)
        return quantized.to(x.device), scale.to(x.device), residual_out.to(x.device)

    def verification(self, *inputs):
        from workloads.numerics import assert_quantized

        def validate(got, expected):
            assert isinstance(got, (tuple, list)) and len(got) == 3
            for actual, target in zip(got, expected, strict=True):
                assert actual.shape == target.shape and actual.dtype == target.dtype
            q, target_q = got[0].cpu(), expected[0].cpu()
            zero = (expected[1].cpu() == 0).expand_as(target_q)
            # A zero-scale row has no rounding ambiguity. Both signed FP8
            # zeros are numerical zero, but every nonzero code must fail.
            assert (q.float()[zero] == 0).all(), "nonzero code at zero scale"
            active = ~zero
            if q.dtype == torch.float8_e4m3fn:
                q = q.view(torch.uint8)[active].view(q.dtype)
                target_q = target_q.view(torch.uint8)[active].view(target_q.dtype)
            else:
                q, target_q = q[active], target_q[active]
            # The shared quantizer policy permits one adjacent code at a rounding tie.
            assert_quantized((q, got[1]), (target_q, expected[1]))
            torch.testing.assert_close(got[2], expected[2], atol=0, rtol=0)

        return Custom(validate, "Full quantized codes, FP32 scales and exact residual sum")
