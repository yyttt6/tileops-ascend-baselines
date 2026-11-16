"""Use SGL's own slot store while preserving the manifest input pools."""
from pathlib import Path


def paged_cache_write(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,UnsupportedCase,local_function,source_hashes
    k,v,kp,vp,slots,ks,vs=case.inputs
    if ks is not None or vs is not None or k.dtype != kp.dtype or v.dtype != vp.dtype:
        raise UnsupportedCase('SGL store is byte/value movement without per-cache FP8 quantization divisors.')
    host='python/sgl_kernel_npu/sgl_kernel_npu/mem_cache/kv_cache_store.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.mem_cache.kv_cache_store','store_kv_cache_prefix_valid_npu_triton',host)
    def run(k,v,kp,vp,slots,ks=None,vs=None):
        outk,outv=kp.clone(),vp.clone()
        # Each input token is a one-element prefix; a negative slot becomes an
        # empty prefix, which the native masked kernel never dereferences.
        commit=(slots>=0).to(torch.int32)
        fn(outk.reshape(-1,*kp.shape[2:]),outv.reshape(-1,*vp.shape[2:]),k.contiguous(),v.contiguous(),slots.reshape(-1,1),commit)
        return outk,outv
    return Candidate(Implementation(run),'sgl_kernel_npu.mem_cache.kv_cache_store.store_kv_cache_prefix_valid_npu_triton',
                     'Clone mutable cache pools per call; flatten pages to slots; map each token to native width-one prefix, negative slot to zero valid length; all untouched rows verified.',
                     source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))
