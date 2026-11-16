"""Exact packed INT4 layout decode, native dequantization and original SGEMMV."""
from pathlib import Path


def unpack(torch, prepacked, table):
    n, packed_k = prepacked.shape
    # Original word format stores four even-K nibbles followed by four odd-K
    # nibbles, with lane-major words. A byte lookup only decodes that storage.
    nibbles = table[prepacked.long()]
    return nibbles.reshape(n, packed_k//64, 4, 4, 2, 4).permute(0,1,3,5,2,4).contiguous().reshape(n,2*packed_k)


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    from .native_fp32_gemv import bind
    from .extra_arithmetic import _affine
    from .additional import mul
    activation, packed, scale, zero = case.inputs
    if activation.dtype not in (torch.float16, torch.bfloat16) or packed.dtype != torch.uint8 or packed.shape[1]%64:
        raise UnsupportedCase('Original W4A16 format requires FP16/BF16 activation and 64-byte aligned packed K')
    product, hashes = bind(activation.device)
    affine, affine_hashes = _affine(case)
    multiply = mul(case)
    table = torch.tensor([[b & 15, b >> 4] for b in range(256)],device=activation.device,dtype=torch.float32)
    def run(activation, packed, scale, zero):
        q = unpack(torch, packed, table)
        group_size = q.shape[1]//scale.shape[1]
        offsets = zero.float().repeat_interleave(group_size,dim=1)
        scales = scale.float().repeat_interleave(group_size,dim=1)
        centered = affine(offsets, q, -1.0)
        weight = multiply.implementation.run(centered,scales).to(activation.dtype)
        # This weight storage cast is explicitly part of the original oracle.
        return product(activation,weight).to(activation.dtype)
    return Candidate(Implementation(run), 'sgl_kernel_npu::mul_add/scale_shift dequantization + full-K sgemmv_shrink',
        'Invert only the declared original INT4 packing through constant byte-nibble lookup and permutation. Original native affine/scale APIs perform (q-zero)*scale in FP32; apply the oracle-mandated activation-dtype weight cast, then original full-K native SGEMMV and final output cast. Independent output tiling only; no partial K reduction, external GEMM or substitute quantization.',
        hashes | affine_hashes | multiply.source_sha256 | source_hashes(Path(__file__)),
        limitations=('W4A16 contract only. W4A8 stays reference-blocked and is not inferred from this separately defined packing/dequantization oracle.',))
