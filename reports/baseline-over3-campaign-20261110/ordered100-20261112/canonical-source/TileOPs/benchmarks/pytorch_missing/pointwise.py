"""Normalization, rotary embedding and complete Hadamard transforms."""

import math

import torch


def rotate(x, trig, layout):
    cosine, sine = trig.float().chunk(2, -1)
    if layout == "neox":
        a, b = x.chunk(2, -1)
        return torch.cat((a * cosine - b * sine, b * cosine + a * sine), -1)
    if layout == "interleaved":
        a, b = x[..., ::2], x[..., 1::2]
        return torch.stack((a * cosine - b * sine, b * cosine + a * sine), -1).flatten(-2)
    raise ValueError(f"Unknown RoPE layout: {layout}")


def fused_qk_norm_rope(
    qkv,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    *,
    num_heads,
    num_kv_heads,
    eps=1e-6,
    rope_layout="neox",
):
    width = q_weight.numel()
    rows = qkv.reshape(qkv.shape[0], num_heads + 2 * num_kv_heads, width)
    trig = cos_sin_cache.index_select(0, positions.long()).float().unsqueeze(1)
    rotary = trig.shape[-1]
    for start, count, weight in ((0, num_heads, q_weight), (num_heads, num_kv_heads, k_weight)):
        value = rows[:, start : start + count].float()
        value = value * (torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float())
        value = torch.cat((rotate(value[..., :rotary], trig, rope_layout), value[..., rotary:]), -1)
        rows[:, start : start + count].copy_(value.to(qkv.dtype))
    return qkv


def fused_add_rms_quant(
    x, residual, weight, *, quant_dtype="float8_e4m3fn", eps=1e-6, scale_ub=None
):
    dtype = getattr(torch, quant_dtype) if isinstance(quant_dtype, str) else quant_dtype
    summed = x.float() + residual.float()
    residual_out = summed.to(x.dtype)
    inv_rms = torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps)
    normalized = ((summed * inv_rms).to(x.dtype) * weight).float()
    maximum = 127.0 if dtype == torch.int8 else torch.finfo(dtype).max
    scale = normalized.abs().amax(-1, keepdim=True) / maximum
    if scale_ub is not None:
        scale = scale.clamp(max=scale_ub)
    codes = normalized / torch.where(scale == 0, torch.ones_like(scale), scale)
    if dtype == torch.int8:
        codes = codes.round()
    minimum = -128.0 if dtype == torch.int8 else -maximum
    return codes.clamp(minimum, maximum).to(dtype), scale, residual_out


def hadamard(x, *, base_order=1):
    n = x.shape[-1]
    if base_order not in (1, 12, 20, 28, 40) or n % base_order:
        raise ValueError("Invalid Hadamard base order or width")
    radix = n // base_order
    if radix < 1 or radix & (radix - 1):
        raise ValueError("Hadamard width/base_order must be a power of two")
    matrix = None
    if base_order != 1:
        # These signs are the mathematical convention, not an execution oracle.
        from workloads.ascend_extended_transform_quant import _HADAMARD_SIGNS

        matrix = torch.tensor(
            [[1.0 if s == "+" else -1.0 for s in row] for row in _HADAMARD_SIGNS[base_order]],
            device=x.device,
        )
    flat = x.reshape(-1, n)
    result = torch.empty_like(flat)
    block = max(1, (1 << 22) // n)
    for start in range(0, flat.shape[0], block):
        value = flat[start : start + block].float().reshape(-1, base_order, radix)
        rows = value.shape[0]
        width = 1
        while width < radix:
            pairs = value.reshape(rows, base_order, -1, 2, width)
            a, b = pairs[..., 0, :], pairs[..., 1, :]
            value = torch.stack((a + b, a - b), -2).reshape(rows, base_order, radix)
            width *= 2
        if matrix is not None:
            value = matrix @ value
        result[start : start + block] = (value.reshape(rows, n) / math.sqrt(n)).to(x.dtype)
    return result.reshape_as(x)
