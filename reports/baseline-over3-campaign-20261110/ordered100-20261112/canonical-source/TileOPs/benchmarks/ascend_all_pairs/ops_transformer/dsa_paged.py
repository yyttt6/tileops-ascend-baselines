"""Original native FP8 G-B dequantization and paged MLA for the 656-byte cache."""
from pathlib import Path


def dsa_paged(case):
    import math,torch
    from types import SimpleNamespace
    from dataclasses import replace
    from .native import GraphCall
    from .adapters import _fp8_block_product
    from .mla import paged
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    from .hashing import source_hashes
    query=case.inputs[0]
    # The inner factory only inspects dtype. It transforms no source input data.
    inner=paged(SimpleNamespace(inputs=(query,query),params={'is_causal':False,'sm_scale':case.params['sm_scale']}),natural_lse=True)
    codec=GraphCall('ops-transformer')
    def run(q,cache,table,cache_lengths,indices):
        batch,sq,heads,dim=q.shape;page=cache.shape[1];chosen=indices.reshape(batch*sq,-1)
        counts=[];selected=[];active=[]
        for i,positions in enumerate(chosen):
            valid=positions[positions!=-1].long();count=valid.numel();counts.append(count)
            if count:
                physical=table[i//sq,valid//page].long()
                selected.append(cache[physical,valid%page]);active.append(i)
        out=torch.zeros((batch*sq,1,heads,512),dtype=q.dtype,device=q.device)
        lse=torch.full((batch*sq,1,heads),float('inf'),dtype=torch.float32,device=q.device)
        if not active:return out.reshape(batch,sq,heads,512),lse.reshape(batch,sq,heads)
        rows=torch.cat(selected).contiguous()
        codes=rows[:,:512].contiguous().view(torch.float8_e4m3fn)
        scales=rows[:,512:528].contiguous().view(torch.float32)
        rope=rows[:,528:656].contiguous().view(torch.bfloat16)
        # Exact identity weights make native G-B GEMM perform only the original
        # FP8 per-token/group dequantization with the required BF16 storage cast.
        identity=torch.eye(512,dtype=torch.float32,device='cpu').to(torch.float8_e4m3fn).to(q.device)
        weight_scales=torch.ones((4,4),dtype=torch.float32,device=q.device)
        latent=_fp8_block_product(codec,codes,identity,scales,weight_scales,torch.bfloat16)
        decoded=torch.cat((latent,rope),-1)
        capacity=-(-max(counts)//128)*128;pages=capacity//128
        packed=torch.zeros((len(active),capacity,dim),dtype=q.dtype,device=q.device)
        offset=0
        for i,source in enumerate(active):
            n=counts[source];packed[i,:n].copy_(decoded[offset:offset+n]);offset+=n
        packed=packed.reshape(len(active)*pages,128,dim)
        block_table=torch.arange(len(active)*pages,dtype=torch.int32,device=q.device).reshape(len(active),pages)
        lengths=torch.tensor([counts[i] for i in active],dtype=torch.int32,device=q.device)
        active_ids=torch.tensor(active,dtype=torch.int64,device=q.device)
        qo=q.reshape(batch*sq,1,heads,dim).index_select(0,active_ids)
        value,log=inner.implementation.run(qo,packed,block_table,lengths)
        out.index_copy_(0,active_ids,value);lse.index_copy_(0,active_ids,log)
        return out.reshape(batch,sq,heads,512),lse.reshape(batch,sq,heads)
    def cleanup():inner.cleanup();codec.cleanup()
    return replace(inner,implementation=Implementation(run),api='ops-transformer::GroupedMatmul+FlashMlaWithKvcache+aclnnFlashMlaWithKvcacheMetadata',adaptation='Reinterpret original 656-byte cache fields after logical sparse gathers, preserving duplicates and -1-only padding. Original FP8 G-B GEMM with identity weights applies four FP32 group scales and the exact BF16 decode rounding. Repack selected rows as native pages; original MLA returns complete O and natural LSE. Empty metadata selections yield the declared zero/+inf constants. No numerical Torch fallback; all decode, metadata transfers, packing and native calls are inside run.',timing_mode='host_roundtrip',cleanup=cleanup,source_sha256=inner.source_sha256|source_hashes(Path(__file__),codec.vendor))
