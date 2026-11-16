"""Lossless FP8 widening and original FLA FP32 final-state matrix products."""
from pathlib import Path


def fp8_matmul(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .cube_matmul import make_matmul
    from .native_affine import make_affine
    matmul,hashes=make_matmul(torch.float32)
    affine,affine_hashes=make_affine()
    p=case.params;batched=case._call.signature.name=='BmmFP8FwdOp'
    dtype=getattr(torch,p['out_dtype'])

    def run(a,b,scale_a,scale_b,bias=None):
        # Both E4M3FN and E5M2 are exact subsets of BF16. Scales remain FP32;
        # scaled operands are never narrowed to BF16.
        a=a.to(torch.bfloat16);b=b.to(torch.bfloat16)
        if batched:
            b=b.transpose(-1,-2) if p.get('trans_b',False) else b
            outputs=[]
            for left,right in zip(a,b,strict=True):
                value=matmul(left,right)
                outputs.append(affine(affine(value,scale_a),scale_b).to(dtype))
            return torch.stack(outputs)
        rows,width=a.shape;columns=b.shape[0]
        if scale_a.numel()==1 and scale_b.numel()==1:
            value=matmul(a,b.t().contiguous())
            value=affine(affine(value,scale_a),scale_b)
        else:
            value=None
            for start in range(0,width,128):
                block=start//128;stop=min(width,start+128)
                left_scale=scale_a.reshape(()) if scale_a.numel()==1 else scale_a[:,block][:,None]
                if scale_b.numel()==1:right_scale=scale_b.reshape(())
                else:
                    right_scale=scale_b[:,block]
                    if right_scale.numel()!=columns:right_scale=right_scale.repeat_interleave(128)[:columns]
                    right_scale=right_scale[None,:]
                product=matmul(a[:,start:stop],b[:,start:stop].t().contiguous())
                product=affine(affine(product,left_scale),right_scale)
                value=product if value is None else affine(product,1.0,value)
        if bias is not None:value=affine(value,1.0,bias.float()[None,:])
        return value.to(dtype)

    return Candidate(Implementation(run),'FLA ChunkFwdH final FP32 state + original native scale/residual, after lossless FP8-to-BF16 widening',
        'Widen each FP8 operand exactly to BF16. Preserve the complete native FP32 matrix result; apply per-tensor scales or separate 128-column reduction-block products with current row/block scales through original FLA FP32 tap/residual APIs. No narrowing of scaled input operands; final output cast once. Handle bias and batched transpose. All transforms inside run; original FP8 input-bit preservation checker retained.',
        hashes|affine_hashes|source_hashes(Path(__file__)))
