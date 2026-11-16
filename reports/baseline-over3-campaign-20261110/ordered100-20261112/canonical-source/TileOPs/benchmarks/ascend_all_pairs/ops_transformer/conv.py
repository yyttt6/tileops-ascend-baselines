"""Preserve accepted pointwise entry; spatial cases use per-call patch layouts."""
from pathlib import Path

def conv2d(case):
    from dataclasses import replace
    from benchmarks.ascend_baselines import source_hashes
    from .hashing import source_hashes
    from benchmarks.ascend_pointwise_conv import transformer_pointwise_conv
    from .adapters import pointwise_conv
    p=case.params
    pair=lambda value:(value,value) if isinstance(value,int) else tuple(value)
    direct=(tuple(case.inputs[1].shape[-2:])==(1,1) and pair(p['stride'])==(1,1) and pair(p['padding'])==(0,0) and p['groups']==1)
    result=(transformer_pointwise_conv if direct else pointwise_conv)(case)
    return replace(result,source_sha256=result.source_sha256|source_hashes(Path(__file__)))
