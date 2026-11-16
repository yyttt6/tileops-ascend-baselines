"""Portable W4A8 v1: low nibble first, signed two's-complement INT4.

This implementation owns its repack format, as permitted by gemm.yaml. It does
not accept vLLM/CUTLASS's private reordered storage or FP8 scale lookup tables.
Group dequantization multiplies the declared scale in FP32, then rounds the
weight to FP8 following upstream d9c2443. The local legacy manifest still uses
BF16 group scales; those inputs are accepted without an implicit scale cast.
FP8 group-scale inputs reproduce the updated upstream semantic contract.
Token, then channel scales apply after the FP32 dot product.
"""

import torch

FORMAT = "tileops-pytorch-w4a8-signed-low-first-fp8-weight-v1"


def repack_w4a8(weight):
    """Pack logical int8 [N,K] weights in [-8,7]; K must be even.

    Values must be valid signed INT4. Range validation belongs to the caller so
    packing itself stays device resident and can be captured by torch.compile.
    """
    if weight.dtype != torch.int8 or weight.ndim != 2 or weight.shape[1] % 2:
        raise ValueError("Expected int8 [N, even K] signed INT4 weights")
    codes = weight.to(torch.int32) & 15
    return (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def gemm_w4a8(
    activation,
    packed_weight,
    weight_group_scale,
    weight_channel_scale,
    activation_scale,
    *,
    group_size=128,
):
    m, k = activation.shape
    n = packed_weight.shape[0]
    if group_size <= 0 or k % group_size or k % 128 or packed_weight.shape != (n, k // 2):
        raise ValueError("Invalid W4A8 shape or group_size")
    if weight_group_scale.shape != (n, k // group_size):
        raise ValueError("Invalid W4A8 group-scale shape")
    result = torch.empty((m, n), dtype=torch.bfloat16, device=activation.device)
    a = activation.float()
    # Bound dequantization scratch independently of the large model dimensions.
    for begin in range(0, n, 512):
        packed = packed_weight[begin : begin + 512].to(torch.int32)
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
        signed = ((codes ^ 8) - 8).float()
        rows = signed.shape[0]
        weight = (
            signed.reshape(rows, k // group_size, group_size)
            * weight_group_scale[begin : begin + rows].float().unsqueeze(-1)
        ).reshape(rows, k)
        weight = weight.to(torch.float8_e4m3fn).float()
        output = (a @ weight.T) * activation_scale.float()[:, None]
        result[:, begin : begin + rows] = (
            output * weight_channel_scale[begin : begin + rows].float()
        ).to(torch.bfloat16)
    return result
