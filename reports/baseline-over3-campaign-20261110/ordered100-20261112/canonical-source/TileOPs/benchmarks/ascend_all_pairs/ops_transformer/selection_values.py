"""Lossless FP32 extrema through original selection indices and value gathers."""
from pathlib import Path


def selection_values(case):
    import math
    import torch
    from dataclasses import replace
    from .native import GraphCall
    from .adapters import _candidate
    from .hashing import source_hashes
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.inputs[0].dtype != torch.float32:
        raise UnsupportedCase('This route binds original FP32 inputs without narrowing.')
    op, p = case._call.signature.name, case.params
    binding = GraphCall('ops-transformer', vendor_name='selectionpairs_transformer')

    def extreme(rows, minimum=False):
        completed = []
        batch = min(65536, max(1024, 1048576 // rows.shape[1]))
        for begin in range(0, len(rows), batch):
            values = rows[begin:begin + batch]
            while values.shape[1] > 1:
                selected = []
                width = 1024 if minimum else 2048
                for start in range(0, values.shape[1], width):
                    source = values[:, start:start + width].contiguous()
                    if source.shape[1] == 1:
                        selected.append(source)
                        continue
                    count = source.shape[1] if minimum else 1
                    scores = torch.empty((len(source), count), dtype=torch.float32, device=source.device)
                    local = torch.empty_like(scores, dtype=torch.int32)
                    aux = torch.empty((0,), dtype=torch.float32, device=source.device)
                    binding.call('MoeGatingTopKSoftmaxV2', [source, None], [scores, local, aux], {'k': count, 'renorm': 1, 'output_softmax_result_flag': False})
                    selected.append(source.gather(1, local[:, -1:].long()))
                    binding.cleanup()
                values = torch.cat(selected, 1)
            completed.append(values[:, 0])
        return torch.cat(completed)

    def pair(a, b, minimum=False):
        a, b = torch.broadcast_tensors(a, b)
        return extreme(torch.stack((a, b), -1).reshape(-1, 2), minimum).reshape(a.shape)

    def run(x, low=None, high=None):
        if op in ('AmaxFwdOp', 'AminFwdOp'):
            dims = p['dim']
            dims = range(x.ndim) if dims is None else ((dims,) if isinstance(dims, int) else dims)
            dims = sorted(d % x.ndim for d in dims)
            kept = [i for i in range(x.ndim) if i not in dims]
            width = math.prod(x.shape[i] for i in dims)
            rows = x.permute(kept + dims).contiguous().reshape(-1, width)
            shape = [1 if i in dims else size for i, size in enumerate(x.shape)] if p['keepdim'] else [x.shape[i] for i in kept]
            return extreme(rows, op == 'AminFwdOp').reshape(shape)
        if op in ('MaximumFwdOp', 'MinimumFwdOp'):
            return pair(x, low, op == 'MinimumFwdOp')
        if op == 'ReluFwdOp':
            return pair(x, torch.zeros((), dtype=x.dtype, device=x.device))
        if op in ('ClampScalarFwdOp', 'HardtanhFwdOp'):
            lo = p.get('min_val') if op == 'HardtanhFwdOp' else p.get('min')
            hi = p.get('max_val') if op == 'HardtanhFwdOp' else p.get('max')
            low = None if lo is None else torch.full((), lo, dtype=x.dtype, device=x.device)
            high = None if hi is None else torch.full((), hi, dtype=x.dtype, device=x.device)
        result = pair(x, low) if low is not None else x
        return pair(result, high, True) if high is not None else result
    result = _candidate(binding, run, 'MoeGatingTopKSoftmaxV2', 'moe/moe_gating_top_k_softmax_v2',
        'Original FP32 V2 renorm=1 selects extreme coordinates before softmax; gather original values rather than normalized scores. Minimum selects the last coordinate of full top-k blocks, then merges original candidates. Broadcast/reduction packing and exact bounds are timed. Original full oracle determines acceptance, including native tie behavior; no external extrema arithmetic.')
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__)))


def extrema_dispatch(case):
    import torch
    from .extrema import extrema
    from .pool_indices import pool_indices
    if case.inputs[0].dtype == torch.float32:
        result = selection_values(case) if case._call.signature.name in ('AmaxFwdOp', 'AminFwdOp') else pool_indices(case)
    else:
        result = extrema(case)
    return _with_dispatch(result)


def pointwise_dispatch(case):
    import torch
    from .pointwise_extrema import pointwise_extrema
    if case.inputs[0].dtype == torch.float32 and case._call.signature.name != 'AbsFwdOp':
        result = selection_values(case)
    else:
        result = pointwise_extrema(case)
    return _with_dispatch(result)


def _with_dispatch(result):
    from dataclasses import replace
    from .hashing import source_hashes
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__)))
