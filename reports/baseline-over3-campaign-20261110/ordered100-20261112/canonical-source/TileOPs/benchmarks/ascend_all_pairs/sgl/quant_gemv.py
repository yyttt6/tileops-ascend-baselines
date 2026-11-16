"""Lossless FP8/INT8 widening with original full-K FP32 SGEMMV and scales."""
from pathlib import Path


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    from .additional import mul
    from .extra_arithmetic import _affine
    from .native_fp32_gemv import bind
    a,b,scale_a,scale_b,*_ = case.inputs
    name = case._call.signature.name
    batched = name == 'BmmFP8FwdOp'
    if name == 'GemmFP8FwdOp' and (scale_a.numel()!=1 or scale_b.numel()!=1):
        raise UnsupportedCase('K-dependent FP8 block scales cannot move outside the reduction; narrowing scaled FP32 operands would be lossy and is not used.')
    if a.dtype not in (torch.int8,torch.float8_e4m3fn) or b.dtype != a.dtype:
        raise UnsupportedCase('This route only widens INT8 or FP8 E4M3 values exactly to FP16.')
    matmul,native_hashes=bind(a.device)
    multiply=mul(case)
    affine,affine_hashes=_affine(case)
    trans_b=case.params.get('trans_b',True)
    output_dtype=getattr(torch,case.params['out_dtype'])
    def run(a,b,scale_a,scale_b,bias=None):
        weights=b if trans_b else b.transpose(-2,-1)
        raw=torch.stack([matmul(aa,bb) for aa,bb in zip(a,weights,strict=True)]) if batched else matmul(a,weights)
        output=multiply.implementation.run(multiply.implementation.run(raw,scale_a.float()),scale_b.float())
        if bias is not None:output=affine(output,bias.float(),1.0)
        return output.to(output_dtype)
    return Candidate(Implementation(run),'sgl_kernel_npu::sgemmv_shrink FP32 output + native scale_shift/mul_add',
        'Losslessly widen FP8 E4M3 or INT8 operands to FP16, retain the complete K reduction in original SGEMMV with FP32 output, then original SGL per-tensor/per-row/per-column scale products and optional bias. Zero padding and independent output partitioning only; final public dtype cast occurs once. No external GEMM or reduction, no narrowed scaled operands.',
        source_hashes(Path(__file__)) | native_hashes | multiply.source_sha256 | affine_hashes,
        limitations=('Original FP32 accumulation and post-reduction scale ordering must pass the unchanged original numerical tolerance; no integer-exact or block-scale equivalence is claimed without that check.',))
