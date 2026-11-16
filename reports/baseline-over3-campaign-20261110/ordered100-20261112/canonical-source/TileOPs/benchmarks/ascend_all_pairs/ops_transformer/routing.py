"""Select original routing APIs according to the manifest scoring function."""
from pathlib import Path


def routing(case):
    if case.params['scoring_func']=='sigmoid':
        from .adapters import fused_topk
        return fused_topk(case)
    import torch
    from dataclasses import replace
    from .native import GraphCall
    from .adapters import _candidate
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
    from .hashing import source_hashes
    p=case.params
    if p['num_expert_group']!=1 or p['topk_group']!=1 or p['routed_scaling_factor']!=1:
        raise UnsupportedCase('Original ungrouped softmax routing is bound only with unit output scaling.')
    binding=GraphCall('ops-transformer')
    def select(logits,k):
        shape=(logits.shape[0],k);scores=torch.empty(shape,dtype=torch.float32,device=logits.device)
        indices=torch.empty(shape,dtype=torch.int32,device=logits.device);rows=torch.empty_like(indices)
        binding.call('MoeGatingTopKSoftmax',[logits.contiguous(),None],[scores,indices,rows],{'k':k})
        return scores,indices
    def run(logits,correction_bias=None):
        if correction_bias is not None:raise UnsupportedCase('Softmax correction-bias routing has no mapped original bias API.')
        logits=logits.float();scores,indices=select(logits,p['top_k'])
        if p['renormalize']:
            selected=logits.gather(1,indices.long())
            scores,permutation=select(selected,p['top_k'])
            indices=indices.gather(1,permutation.long())
        return scores,indices
    result=_candidate(binding,run,'MoeGatingTopKSoftmax','moe/moe_gating_top_k_softmax','Original FP32 softmax/top-k selects experts. For renormalization, gather their original logits and call the same original API over precisely the selected set; restore expert indices through its returned permutation. All casts/gathers/native calls are timed; no external score arithmetic.')
    return replace(result,source_sha256=result.source_sha256|source_hashes(Path(__file__)))
