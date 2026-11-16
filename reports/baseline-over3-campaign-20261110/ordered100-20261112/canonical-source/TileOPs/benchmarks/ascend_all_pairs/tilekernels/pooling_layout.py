"""Add the omitted batch dimension for unbatched original pooling contracts."""
from .pooling import pooling


def pooling_layout(case):
    import re
    from types import SimpleNamespace
    name = case._call.signature.name
    if name == 'MeanPoolingFwdOp': return pooling(case)
    spatial_rank = int(re.search(r'([123])d', name).group(1))
    x = case.inputs[0]
    if x.ndim != spatial_rank + 1: return pooling(case)
    tensors = dict(case._call.tensors)
    for key in ('input', 'output', 'indices'):
        if key in tensors:
            shape, dtype = tensors[key]
            tensors[key] = ((1, *shape), dtype)
    proxy = SimpleNamespace(_call=SimpleNamespace(signature=case._call.signature, tensors=tensors),
                            params=case.params, inputs=(x.unsqueeze(0),))
    inner = pooling(proxy)
    def run(x):
        out = inner.implementation.run(x.unsqueeze(0))
        return tuple(t.squeeze(0) for t in out) if isinstance(out, tuple) else out.squeeze(0)
    from dataclasses import replace
    from pathlib import Path
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    return replace(inner, implementation=Implementation(run),
                   adaptation='Add/remove singleton batch dimension on every unbatched call. ' + inner.adaptation,
                   source_sha256=inner.source_sha256 | source_hashes(Path(__file__)))
