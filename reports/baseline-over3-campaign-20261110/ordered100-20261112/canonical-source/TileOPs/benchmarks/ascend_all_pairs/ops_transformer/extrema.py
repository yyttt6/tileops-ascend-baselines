"""Additional layout routes through original FlashAttentionScore raw maxima.

Kept separate until registered and device-validated. All packing and the unused
native attention outputs remain in the measured call.
"""
from pathlib import Path


def extrema(case):
    import itertools
    import math
    import torch
    from collections import defaultdict
    from dataclasses import replace
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_transformer_graph import _attention
    from benchmarks.ascend_baselines import UnsupportedCase
    from .adapters import _candidate, _gemm
    from .hashing import source_hashes

    op = case._call.signature.name
    p = case.params
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16):
        raise UnsupportedCase('Original FlashAttentionScore basis-query route requires FP16/BF16; no FP32 narrowing.')
    minimum = op == 'AminFwdOp'
    binding = GraphCall('ops-transformer')

    def row_extreme(rows):
        width = rows.shape[-1]
        # Limit the padded key buffer; every block is still a complete reduction.
        block = min(32768, max(1, (1 << 25) // (128 * width)))
        values = []
        for start in range(0, rows.shape[0], block):
            source = rows[start:start + block]
            keys = torch.nn.functional.pad(source[:, :, None, None], (0, 127)).contiguous()
            query = torch.zeros((source.shape[0], 1, 1, 128), dtype=rows.dtype, device=rows.device)
            query[..., 0] = -1 if minimum else 1
            _, maxima = _attention(binding, query, keys, keys, 1.)
            values.append(maxima[:, 0, 0, 0].to(rows.dtype))
        result = torch.cat(values)
        if minimum:
            coefficient = torch.full((1, 1), -1., dtype=rows.dtype, device=rows.device)
            result = _gemm(binding, result[:, None], coefficient, rows.dtype)[:, 0]
        return result

    def run(x):
        if op in ('AmaxFwdOp', 'AminFwdOp'):
            dims = p['dim']
            dims = range(x.ndim) if dims is None else ((dims,) if isinstance(dims, int) else dims)
            dims = sorted(d % x.ndim for d in dims)
            kept = [i for i in range(x.ndim) if i not in dims]
            width = math.prod(x.shape[i] for i in dims)
            rows = x.permute(kept + dims).contiguous().reshape(-1, width)
            shape = [1 if i in dims else size for i, size in enumerate(x.shape)] if p['keepdim'] else [x.shape[i] for i in kept]
            return row_extreme(rows).reshape(shape)
        if op == 'AdaptiveMaxPool2dFwdOp':
            target = p['output_size']
            target = (target, target) if isinstance(target, int) else target
            h, w = x.shape[-2:]
            groups = defaultdict(list)
            for i, j in itertools.product(range(target[0]), range(target[1])):
                bounds = (i * h // target[0], -(-((i + 1) * h) // target[0]),
                          j * w // target[1], -(-((j + 1) * w) // target[1]))
                groups[(bounds[1] - bounds[0]) * (bounds[3] - bounds[2])].append((i * target[1] + j, bounds))
            leading = x.shape[:-2]
            out = torch.empty((*leading, math.prod(target)), dtype=x.dtype, device=x.device)
            for width, items in groups.items():
                rows = torch.stack([x[..., a:b, c:d].contiguous().flatten(-2)
                                    for _, (a, b, c, d) in items], -2)
                values = row_extreme(rows.reshape(-1, width)).reshape(*leading, len(items))
                ids = torch.tensor([i for i, _ in items], dtype=torch.int64, device=x.device)
                out.index_copy_(-1, ids, values)
            return out.reshape(*leading, *target)
        rank = {'MaxPool1dFwdOp': 1, 'MaxPool2dFwdOp': 2, 'MaxPool3dFwdOp': 3}[op]
        leading = x.shape[:-rank]
        tup = lambda value: (value,) * rank if isinstance(value, int) else tuple(value)
        kernel = tup(p['kernel_size'])
        stride = tup(p['stride'] if p['stride'] is not None else p['kernel_size'])
        padding, dilation = tup(p['padding']), tup(p['dilation'])
        output, right = [], []
        for size, k, s, pad, dil in zip(x.shape[-rank:], kernel, stride, padding, dilation, strict=True):
            effective = dil * (k - 1) + 1
            count = (size + 2 * pad - effective + (s - 1 if p['ceil_mode'] else 0)) // s + 1
            if p['ceil_mode'] and (count - 1) * s >= size + pad:
                count -= 1
            output.append(count)
            right.append(max(pad, (count - 1) * s + effective - size - pad))
        pads = tuple(v for left, end in reversed(list(zip(padding, right))) for v in (left, end))
        padded = torch.nn.functional.pad(x, pads, value=float('-inf'))
        sizes = (*leading, *output, *kernel)
        strides = (*padded.stride()[:-rank],
                   *[v * s for v, s in zip(padded.stride()[-rank:], stride)],
                   *[v * d for v, d in zip(padded.stride()[-rank:], dilation)])
        rows = padded.as_strided(sizes, strides).contiguous().reshape(-1, math.prod(kernel))
        return row_extreme(rows).reshape(*leading, *output)

    result = _candidate(binding, run, 'FlashAttentionScore'+('+GroupedMatmulAdd' if minimum else ''),
                        'attention/flash_attention_score',
                        'Original attention raw maximum with a signed basis query; complete reduction/window packing, exact -inf padding, stride/dilation/ceil/adaptive bounds and output views. Minimum additionally uses native GEMM with constant -1. All native outputs and layout work are included; no Torch reduction/pooling arithmetic.')
    if minimum:
        from benchmarks.ascend_baselines import WORKSPACE
        result = replace(result, source_sha256=result.source_sha256 | source_hashes(
            WORKSPACE/'ascend_baselines/ops-transformer/gmm/grouped_matmul_add'))
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__)))
