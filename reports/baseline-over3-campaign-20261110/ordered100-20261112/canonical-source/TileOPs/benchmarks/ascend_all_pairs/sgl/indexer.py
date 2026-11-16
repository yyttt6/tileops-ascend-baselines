"""Select complete original native index outputs; no framework top-k fallback."""
from pathlib import Path


def topk_select(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes
    host='python/sgl_kernel_npu/sgl_kernel_npu/indexer/flash_block_score_prefill.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.indexer.flash_block_score_prefill','flash_prefill_bnsd_topk_from_score',host)
    count=case.params['topk']
    def run(scores,starts,ends):
        b,s,n,g=scores.shape
        positions=torch.arange(n,device=scores.device)[None,None,:,None]
        valid=(positions>=starts[:,:,None,None])&(positions<ends[:,:,None,None])
        packed=scores.masked_fill(~valid,float('-inf')).permute(3,0,1,2).contiguous().reshape(g,b*s,n)
        # Complete input score row, block_size=1: native top-k ranks exactly the
        # original values. Native extra local-block column is outside contract.
        selected=fn(packed,ends.reshape(-1).to(torch.int32),count,1)[...,:count]
        selected=torch.where(selected<0,n,selected)
        return selected.permute(1,0,2).reshape(b,s,g,count).contiguous()
    return Candidate(Implementation(run),'sgl_kernel_npu.indexer.flash_block_score_prefill.flash_prefill_bnsd_topk_from_score',
        'Broadcast supplied integer valid windows to pad excluded score locations with -inf; reshape complete K rows into original API, block_size=1; crop optional local-block append and map native -1 sentinel to manifest K.',
        source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host),
        limitations=('Original full-width repeated-selection schedule unchanged; large prefill workloads may exceed execution budget and remain resource blocked, not mathematically unsupported.',))
