"""Boolean reductions by exact native min/max selection of Boolean values."""
from pathlib import Path


def boolean_reduce(case):
    import math
    import torch
    from dataclasses import replace
    from .native import GraphCall
    from .adapters import _candidate
    from .hashing import source_hashes
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.inputs[0].dtype != torch.bool:
        raise UnsupportedCase('This exact native selector binds Boolean data only; arbitrary numeric truth conversion is not a lossless dtype adapter.')
    minimum = case._call.signature.name == 'AllFwdOp'
    p = case.params
    binding = GraphCall('ops-transformer', vendor_name='selectionpairs_transformer')
    def run(x):
        dims = p['dim']
        dims = range(x.ndim) if dims is None else ((dims,) if isinstance(dims, int) else dims)
        dims = sorted(d % x.ndim for d in dims)
        kept = [i for i in range(x.ndim) if i not in dims]
        width = math.prod(x.shape[i] for i in dims)
        rows = x.permute(kept + dims).contiguous().reshape(-1, width)
        result = []
        for begin in range(0, len(rows), 1024):
            values = rows[begin:begin + 1024]
            while values.shape[1] > 1:
                picked = []
                for left in range(0, values.shape[1], 1024):
                    original = values[:, left:left + 1024]
                    source = original.float().contiguous()
                    k = source.shape[1] if minimum else 1
                    scores = torch.empty((len(source), k), dtype=torch.float32, device=x.device)
                    ids = torch.empty_like(scores, dtype=torch.int32)
                    full = torch.empty((0,), dtype=torch.float32, device=x.device)
                    binding.call('MoeGatingTopKSoftmaxV2', [source, None], [scores, ids, full],
                                 {'k': k, 'renorm': 1, 'output_softmax_result_flag': False})
                    picked.append(original.gather(1, ids[:, -1:].long()))
                    binding.cleanup()
                values = torch.cat(picked, 1)
            result.append(values[:, 0])
        shape = [1 if i in dims else size for i, size in enumerate(x.shape)] if p['keepdim'] else [x.shape[i] for i in kept]
        return torch.cat(result).reshape(shape)
    candidate = _candidate(binding, run, 'MoeGatingTopKSoftmaxV2', 'moe/moe_gating_top_k_softmax_v2',
        'Boolean false/true widen exactly to FP32 zero/one for original raw-score native top-k. Select native minimum for All or maximum for Any, gathering original Boolean values. Reduction-axis layout and hierarchical selection preserve exact Boolean semantics; no external Boolean reduction or numeric truth conversion.')
    return replace(candidate, source_sha256=candidate.source_sha256 | source_hashes(Path(__file__)))
