"""Native top-k indices establish a threshold; native selectors preserve ties."""
from pathlib import Path


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,local_function,source_hashes,LIBRARIES
    thost='python/sgl_kernel_npu/sgl_kernel_npu/indexer/flash_block_score_prefill.py'
    ahost='python/sgl_kernel_npu/sgl_kernel_npu/sample/argmax_softmax_prob.py'
    topk=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.indexer.flash_block_score_prefill','flash_prefill_bnsd_topk_from_score',thost)
    argmax=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.sample.argmax_softmax_prob','argmax_softmax_prob_fused',ahost)
    def run(logits,k):
        vocab=logits.shape[-1]
        counts=k.cpu().tolist();outputs=[]
        for row,count in zip(logits,counts):
            if count>=vocab:
                outputs.append(row.clone());continue
            native_count=1<<(int(count)-1).bit_length()
            indices=topk(row.float().reshape(1,1,vocab),torch.tensor([vocab],device=row.device,dtype=torch.int32),native_count,1)[0,0,:count]
            threshold=row[indices[-1].long()]
            # First-index ties mean [logit,threshold] chooses index1 only
            # when logit is strictly below the kth value; all ties survive.
            flags=argmax(torch.stack((row,threshold.expand_as(row)),dim=-1))[0]
            payload=torch.stack((row,torch.full_like(row,float('-inf'))),dim=-1)
            outputs.append(payload.gather(1,flags[:,None]).reshape_as(row))
        return torch.stack(outputs)
    return Candidate(Implementation(run),'sgl_kernel_npu::flash_prefill_bnsd_topk_from_score + argmax_softmax_prob_fused',
        'Read actual per-row k, pad native selection count to its power-of-two requirement and crop first k ranks; original full-row top-k establishes kth threshold. Original first-tie pair argmax marks strict-below entries, gather original logits or -inf without renormalization, preserving every retained bit and boundary tie.',
        source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/thost,LIBRARIES/'sgl-kernel-npu'/ahost),timing_mode='host_roundtrip',
        limitations=('Native top-k host uses a full power-of-two vocabulary row; compiler resource limits must be measured, never inferred solely from example shapes.',))
