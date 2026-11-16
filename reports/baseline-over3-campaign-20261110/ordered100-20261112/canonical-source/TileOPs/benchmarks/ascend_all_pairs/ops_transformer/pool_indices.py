"""Exact pool values and original spatial indices via the native top-1 API."""
from pathlib import Path


def pool_indices(case):
    import itertools
    import math
    import torch
    from collections import defaultdict
    from dataclasses import replace
    from .native import GraphCall
    from .adapters import _candidate
    from .hashing import source_hashes
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase('Native top-1 supports exact floating-to-FP32 widening only.')
    binding = GraphCall('ops-transformer', vendor_name='selectionpairs_transformer')
    p, op = case.params, case._call.signature.name
    with_indices = 'Indices' in op
    if not with_indices:
        op = op.replace('FwdOp', 'IndicesFwdOp')

    def select(rows, coordinates):
        all_values, all_ids = [], []
        batch = min(65536, max(1024, 1048576 // rows.shape[1]))
        for begin in range(0, len(rows), batch):
            source = rows[begin:begin + batch]
            scores = source.float()
            ids = coordinates[begin:begin + batch]
            values = source
            while True:
                chosen_values, chosen_ids = [], []
                for left in range(0, scores.shape[1], 2048):
                    x = scores[:, left:left + 2048].contiguous()
                    prob = torch.empty((len(x), 1), dtype=torch.float32, device=x.device)
                    loc = torch.empty_like(prob, dtype=torch.int32)
                    aux = torch.empty((0,), dtype=torch.float32, device=x.device)
                    binding.call('MoeGatingTopKSoftmaxV2', [x, None], [prob, loc, aux], {'k': 1, 'renorm': 1, 'output_softmax_result_flag': False})
                    local = loc.long()
                    chosen_values.append(values[:, left:left + 2048].gather(1, local))
                    chosen_ids.append(ids[:, left:left + 2048].gather(1, local))
                    binding.cleanup()
                values = torch.cat(chosen_values, 1)
                ids = torch.cat(chosen_ids, 1)
                if values.shape[1] == 1:
                    all_values.append(values[:, 0]); all_ids.append(ids[:, 0])
                    break
                scores = values.float()
        return torch.cat(all_values), torch.cat(all_ids)

    def run(x):
        if op == 'AdaptiveMaxPool2dIndicesFwdOp':
            target = p['output_size']
            target = (target, target) if isinstance(target, int) else target
            h, w = x.shape[-2:]
            leading = x.shape[:-2]
            spatial = torch.arange(h * w, device=x.device, dtype=torch.int64).reshape(h, w)
            groups = defaultdict(list)
            for i, j in itertools.product(range(target[0]), range(target[1])):
                bounds = (i * h // target[0], -(-((i + 1) * h) // target[0]),
                          j * w // target[1], -(-((j + 1) * w) // target[1]))
                groups[(bounds[1] - bounds[0]) * (bounds[3] - bounds[2])].append((i * target[1] + j, bounds))
            out = torch.empty((*leading, math.prod(target)), dtype=x.dtype, device=x.device)
            indices = torch.empty_like(out, dtype=torch.int64)
            for width, items in groups.items():
                rows = torch.stack([x[..., a:b, c:d].contiguous().flatten(-2)
                                    for _, (a, b, c, d) in items], -2)
                coords = torch.stack([spatial[a:b, c:d].contiguous().flatten()
                                      for _, (a, b, c, d) in items], 0).expand(*leading, len(items), width)
                values, positions = select(rows.reshape(-1, width), coords.reshape(-1, width))
                ids = torch.tensor([i for i, _ in items], dtype=torch.int64, device=x.device)
                out.index_copy_(-1, ids, values.reshape(*leading, len(items)))
                indices.index_copy_(-1, ids, positions.reshape(*leading, len(items)))
            result = out.reshape(*leading, *target)
            return (result, indices.reshape(*leading, *target)) if with_indices else result
        rank = {'MaxPool1dIndicesFwdOp': 1, 'MaxPool2dIndicesFwdOp': 2, 'MaxPool3dIndicesFwdOp': 3}[op]
        leading, sizes = x.shape[:-rank], x.shape[-rank:]
        tup = lambda value: (value,) * rank if isinstance(value, int) else tuple(value)
        kernel = tup(p['kernel_size'])
        stride = tup(p['stride'] if p['stride'] is not None else p['kernel_size'])
        padding, dilation = tup(p['padding']), tup(p['dilation'])
        output, right = [], []
        for size, k, s, pad, dil in zip(sizes, kernel, stride, padding, dilation, strict=True):
            effective = dil * (k - 1) + 1
            count = (size + 2 * pad - effective + (s - 1 if p['ceil_mode'] else 0)) // s + 1
            if p['ceil_mode'] and (count - 1) * s >= size + pad:
                count -= 1
            output.append(count)
            right.append(max(pad, (count - 1) * s + effective - size - pad))
        pads = tuple(v for left, end in reversed(list(zip(padding, right))) for v in (left, end))
        padded = torch.nn.functional.pad(x, pads, value=float('-inf'))
        spatial = torch.arange(math.prod(sizes), device=x.device, dtype=torch.int64).reshape(sizes)
        spatial = torch.nn.functional.pad(spatial, pads, value=-1)
        def windows(tensor, prefix):
            strides = tensor.stride()[-rank:]
            return tensor.as_strided((*prefix, *output, *kernel),
                (*tensor.stride()[:-rank], *[v * s for v, s in zip(strides, stride)],
                 *[v * d for v, d in zip(strides, dilation)]))
        width = math.prod(kernel)
        rows = windows(padded, leading).contiguous().reshape(-1, width)
        coords = windows(spatial, ()).contiguous().reshape(*output, width)
        coords = coords.expand(*leading, *output, width).reshape(-1, width)
        values, indices = select(rows, coords)
        result = values.reshape(*leading, *output)
        return (result, indices.reshape(*leading, *output)) if with_indices else result

    result = _candidate(binding, run, 'MoeGatingTopKSoftmaxV2', 'moe/moe_gating_top_k_softmax_v2',
        'Pack exact padded/dilated/ceil/adaptive windows and original flattened spatial-index metadata. Original V2 renorm=1 raw-score top-1 selects each maximum; gather values from original input and indices from metadata. Full exact pool oracle checks both outputs and first-occurrence ties; no Torch max or argmax calculation.')
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__)))
