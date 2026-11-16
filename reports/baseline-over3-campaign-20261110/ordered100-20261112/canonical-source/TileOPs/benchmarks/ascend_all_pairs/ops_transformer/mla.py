"""Native paged MLA plus original-library FP32 log-unit conversion."""
from pathlib import Path

def paged(case, *, natural_lse=False):
    import ctypes as C
    import math,torch
    from dataclasses import replace
    from .adapters import _candidate,_float_product,DISPATCH_EVENTS
    from .native import GraphCall
    from benchmarks.ascend_math_more import _MathCall
    from benchmarks.ascend_baselines import UnsupportedCase,WORKSPACE,source_hashes
    from .hashing import source_hashes
    q,cache,*_=case.inputs
    if q.dtype!=cache.dtype or q.dtype not in (torch.float16,torch.bfloat16):
        raise UnsupportedCase('Original FlashMlaWithKvcache is nonquantized; scaled FP8 cache contract has no direct original API.')
    binding=GraphCall('ops-transformer','mlapairs_transformer');product=GraphCall('ops-transformer')
    binary=binding.vendor/'op_api/lib/libcust_opapi.so'
    api='aclnnFlashMlaWithKvcacheMetadata'
    metadata=_MathCall(api,[C.c_void_p]*3+[C.c_int64]*7+[C.c_char_p,C.c_void_p],library='ops-transformer',host_library=binary)
    p=case.params
    def run(q,cache,table,lengths,scale=None):
        batch,sq,heads,dim=q.shape;capacity=table.shape[1]*cache.shape[1]
        # The original native page size is 128. Repack the manifest's complete
        # logical pages, including arbitrary physical page-table permutations.
        gathered=cache[table.long()].reshape(batch,capacity,dim)
        pad=-capacity%128
        if pad:gathered=torch.nn.functional.pad(gathered,(0,0,0,pad))
        new_pages=(capacity+pad)//128
        packed=gathered.reshape(batch*new_pages,128,1,dim).contiguous()
        new_table=torch.arange(batch*new_pages,dtype=torch.int32,device=q.device).reshape(batch,new_pages)
        cu=torch.arange(batch+1,dtype=torch.int32,device=q.device)*sq
        query=q.reshape(batch*sq,heads,dim).contiguous();mode=3 if p['is_causal'] else 0
        mask=(torch.arange(2048,device=q.device)[None,:]>torch.arange(2048,device=q.device)[:,None]).to(torch.int8) if mode else None
        size=(((72+144)*batch+1)*16+4095)//4096*4096
        meta=torch.empty(size,dtype=torch.int32,device=q.device);record=metadata.record()
        descriptors=[metadata.tensor(x,record) for x in (cu,lengths)]
        desc=metadata.tensor(meta,record)
        metadata.call([*descriptors,None,sq,capacity,heads,1,dim,512,mode,b'TND',desc],record,q.device)
        DISPATCH_EVENTS.append({'op':api,'source':str(WORKSPACE/'ascend_baselines/ops-transformer/attention/flash_mla_with_kvcache_metadata/op_host/op_api/aclnn_flash_mla_with_kvcache_metadata.cpp'),'binary':str(binary),'dispatch_owner_checked':True,'returned':0})
        out=torch.empty((batch*sq,heads,512),dtype=q.dtype,device=q.device)
        lse=torch.empty((heads,batch*sq),dtype=torch.float32,device=q.device)
        binding.call('FlashMlaWithKvcache',[query,packed,new_table,lengths,cu,None,mask,meta],[out,lse],{'head_dim_v':512,'softmax_scale':float(p.get('sm_scale') or dim**-.5),'mask_mode':mode,'max_seqlen_q':sq,'max_seqlen_kv':capacity,'layout_q':'TND','layout_kv':'PA_BBND','layout_out':'TND','return_softmax_lse':1})
        if natural_lse:return out.reshape(batch,sq,heads,512),lse.T.contiguous().reshape(batch,sq,heads)
        # Native returns natural LSE; manifest requires base-2 LSE. Arithmetic
        # executes in the original FP32 GroupedMatmul, never a torch fallback.
        coefficient=torch.full((1,1),1/math.log(2),dtype=torch.float32,device=q.device)
        log2=_float_product(product,lse.T.contiguous().reshape(-1,1),coefficient)
        return out.reshape(batch,sq,heads,512),log2.reshape(batch,sq,heads)
    result=_candidate(binding,run,'FlashMlaWithKvcache+aclnnFlashMlaWithKvcacheMetadata+GroupedMatmul','attention/flash_mla_with_kvcache','Per-call logical-page gather and exact 64-to-128 repack, original native metadata and MLA attention with complete O/LSE; original FP32 GEMM changes natural-log units to manifest base-2. All input-dependent work timed, no numerical torch fallback.')
    if natural_lse:result=replace(result,adaptation='Per-call page gather/repack and original native MLA metadata/attention; return complete natural-log LSE without log-unit conversion.')
    cleanup_original=result.cleanup
    def cleanup():cleanup_original();metadata.cleanup();product.cleanup()
    return replace(result,cleanup=cleanup,source_sha256=result.source_sha256|source_hashes(Path(__file__),product.vendor,WORKSPACE/'ascend_baselines/ops-transformer/gmm/grouped_matmul',WORKSPACE/'ascend_baselines/ops-transformer/attention/flash_mla_with_kvcache_metadata',WORKSPACE/'TileOPs/benchmarks/ascend_math_more.py'))
