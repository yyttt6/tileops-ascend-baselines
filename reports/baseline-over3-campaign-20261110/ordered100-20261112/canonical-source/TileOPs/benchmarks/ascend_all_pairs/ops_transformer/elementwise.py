"""Recover complete native activation by inverting a full selection permutation."""
from pathlib import Path

def sigmoid(case):
    import torch
    from dataclasses import replace
    from .native import GraphCall
    from .adapters import _candidate
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
    from .hashing import source_hashes
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise UnsupportedCase('Original MoeFusedTopk has floating activation inputs; no lossy input conversion.')
    binding=GraphCall('ops-transformer')
    def run(x):
        size=x.numel();width=256
        packed=torch.nn.functional.pad(x.float().contiguous().flatten(),(0,-size%width)).reshape(-1,width)
        bias=torch.zeros(width,dtype=torch.float32,device=x.device)
        values=torch.empty_like(packed);indices=torch.empty(packed.shape,dtype=torch.int32,device=x.device)
        binding.call('MoeFusedTopk',[packed,bias,None,None],[values,indices],{'group_num':1,'group_topk':1,'top_n':1,'top_k':width,'activate_type':0,'is_norm':False,'scale':1.,'enable_expert_mapping':False})
        restored=torch.empty_like(values).scatter_(1,indices.long(),values)
        return restored.flatten()[:size].reshape_as(x).to(x.dtype)
    result=_candidate(binding,run,'MoeFusedTopk','moe/moe_fused_topk','Flatten and zero-pad rows of 256, exact FP32 widening, select every native sigmoid coordinate without normalization or rescaling, invert native permutation and crop. No activation arithmetic in adapter; all transforms timed.')
    return replace(result,source_sha256=result.source_sha256|source_hashes(Path(__file__)))
