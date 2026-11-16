"""RMS QK normalization and supplied-table RoPE through original FLA APIs."""
from math import sqrt
from pathlib import Path


def fused_qk(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .providers import _triton_module
    from .native_affine import make_affine
    norm,source=_triton_module('l2norm')
    affine,hashes=make_affine()
    p=case.params

    def run(qkv,q_weight,k_weight,cos_sin_cache,positions):
        width=q_weight.numel();heads=p['num_heads'];kv_heads=p['num_kv_heads']
        result=qkv.clone().reshape(qkv.shape[0],heads+2*kv_heads,width)
        cache=cos_sin_cache[positions.long()].float()
        rotary=cache.shape[-1]
        cos,sin=(part[:,None,:] for part in cache.chunk(2,dim=-1))
        for start,count,weight in ((0,heads,q_weight),(heads,kv_heads,k_weight)):
            data=result[:,start:start+count].contiguous().float()
            value=norm.l2norm_fwd(data,eps=width*p['eps'],output_dtype=torch.float32)[0]
            value=affine(value,affine(weight.float(),sqrt(width)))
            rotated=value[...,:rotary]
            if p['rope_layout']=='neox':first,second=rotated.chunk(2,dim=-1)
            else:first,second=rotated[...,::2],rotated[...,1::2]
            negative_sin=affine(sin,-1.0)
            left=affine(second,negative_sin,affine(first,cos))
            right=affine(first,sin,affine(second,cos))
            rotated=torch.cat((left,right),dim=-1) if p['rope_layout']=='neox' else torch.stack((left,right),dim=-1).flatten(-2)
            value[...,:rotary]=rotated
            result[:,start:start+count]=value.to(qkv.dtype)
        return result.reshape(qkv.shape)

    return Candidate(Implementation(run),'FLA l2norm_fwd + causal_conv1d_fwd_impl_old(QK RMS and supplied-cache rotation)',
        'Preserve FP32 normalized values through rotation; gather the supplied cache using current positions, support partial rotary dimensions and both pair layouts. All norm/product/addition arithmetic is original FLA; return the complete private QKV with V bitwise unchanged. All input-dependent transforms inside run.',
        hashes|source_hashes(Path(__file__),source,source.with_name('utils.py')))
