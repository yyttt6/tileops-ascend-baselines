"""SGL native slot store into disjoint latent and rotary cache views."""
from pathlib import Path


def write(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,local_function,source_hashes,LIBRARIES,UnsupportedCase
    latent,positional,cache,*_=case.inputs
    if cache.dtype!=latent.dtype or cache.dtype!=positional.dtype:
        raise UnsupportedCase('Original SGL slot store has no FP8 quantization/divisor; only matching storage dtypes are supported.')
    shost='python/sgl_kernel_npu/sgl_kernel_npu/mem_cache/kv_cache_store.py'
    rhost='python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_rope_qk_mqa.py'
    store=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.mem_cache.kv_cache_store','store_kv_cache_prefix_valid_npu_triton',shost)
    rotate=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.norm.fused_rope_qk_mqa','fused_rope_qk_mqa',rhost)
    rank,tail=latent.shape[-1],positional.shape[-1]
    def run(latent,positional,cache,slots,scale=None,positions=None,table=None):
        if scale is not None:raise UnsupportedCase('A supplied cache conversion scale has no native slot-store parameter.')
        tokens=(slots>=0).nonzero().reshape(-1)
        ll=latent[tokens].contiguous();pp=positional[tokens].contiguous()
        if case.params['fuse_rope']:
            trig=table[positions[tokens].long()].float().contiguous()
            widened=pp.float().unsqueeze(1).contiguous()
            pp=rotate(widened,widened,trig,tail,case.params['rope_layout']=='neox')[0][:,0].to(positional.dtype)
        output=cache.clone().contiguous()
        flat=output.reshape(-1,rank+tail)
        # Head dimension is one, so its stride can equal the actual subrow
        # width while the physical row stride retains the full concatenation.
        kk=flat[:,:rank].as_strided((flat.shape[0],1,rank),(rank+tail,rank,1))
        vv=flat[:,rank:].as_strided((flat.shape[0],1,tail),(rank+tail,tail,1))
        store(kk,vv,ll[:,None],pp[:,None],slots[tokens,None],torch.ones(tokens.numel(),device=slots.device,dtype=torch.int32))
        return output
    return Candidate(Implementation(run),'sgl_kernel_npu::store_kv_cache_prefix_valid_npu_triton + fused_rope_qk_mqa',
        'Clone complete cache; disjoint strided latent/PE views retain full physical row stride. Gather valid slots and optional native FP32 RoPE, restore original PE dtype, native store writes both regions; every untouched cache bit preserved.',
        source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/shost,LIBRARIES/'sgl-kernel-npu'/rhost))
