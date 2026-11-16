"""Native top-1 hierarchy; complete original index and tie oracle is retained."""
from pathlib import Path


def arg_indices(case):
    import math
    import torch
    from dataclasses import replace
    from .native import GraphCall
    from benchmarks.ascend_graph_call import GraphCall as MatrixCall
    from benchmarks.ascend_baselines import UnsupportedCase, WORKSPACE
    from .adapters import _candidate, _gemm
    from .hashing import source_hashes
    minimum = case._call.signature.name == 'ArgminFwdOp'
    dtype = case.inputs[0].dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase('Native top-1 requires exact floating inputs or lossless FP32 widening.')
    binding = GraphCall('ops-transformer', vendor_name='selectionpairs_transformer')
    full_minimum = minimum and dtype == torch.float32
    matrix = (GraphCall('ops-transformer') if full_minimum else MatrixCall('ops-transformer')) if minimum else None
    dim, keep = case.params['dim'], case.params['keepdim']

    def select(rows):
        width = rows.shape[-1]
        ids = torch.arange(width, dtype=torch.int64, device=rows.device)[None].expand(len(rows), -1)
        while True:
            values, positions = [], []
            for begin in range(0, rows.shape[1], 2048):
                x = rows[:, begin:begin + 2048].contiguous()
                prob = torch.empty((len(rows), 1), dtype=torch.float32, device=rows.device)
                local = torch.empty_like(prob, dtype=torch.int32)
                row_ids = torch.empty((0,), dtype=torch.float32, device=rows.device)
                binding.call('MoeGatingTopKSoftmaxV2', [x, None], [prob, local, row_ids], {'k': 1, 'renorm': 1, 'output_softmax_result_flag': False})
                selected = local.long()
                values.append(x.gather(1, selected))
                positions.append(ids[:, begin:begin + 2048].gather(1, selected))
                binding.cleanup()
            rows, ids = torch.cat(values, 1), torch.cat(positions, 1)
            if rows.shape[1] == 1:
                return ids[:, 0]

    def run(x):
        if dim is None:
            rows, shape = x.contiguous().reshape(1, -1), [1] * x.ndim if keep else []
        else:
            axis = dim % x.ndim
            packed = x.movedim(axis, -1).contiguous()
            rows = packed.reshape(-1, packed.shape[-1])
            shape = [1 if i == axis else v for i, v in enumerate(x.shape)] if keep else [v for i, v in enumerate(x.shape) if i != axis]
        if minimum:
            coefficient = torch.full((1, 1), -1., dtype=dtype, device=x.device)
            if full_minimum:
                from .routing_arithmetic import weighted_rows
                rows = weighted_rows(matrix, rows.reshape(-1, 1, 1), coefficient.expand(rows.numel(), 1).contiguous()).reshape(rows.shape)
            else:
                rows = _gemm(matrix, rows.reshape(-1, 1), coefficient, dtype).reshape(rows.shape)
            matrix.cleanup()
        rows = rows.float()
        selected = [select(rows[start:start + 1024]) for start in range(0, len(rows), 1024)]
        return torch.cat(selected).reshape(shape)

    result = _candidate(binding, run, 'MoeGatingTopKSoftmaxV2' + ('+MoeFinalizeRoutingV2' if full_minimum else '+GroupedMatmulAdd' if minimum else ''),
                        'moe/moe_gating_top_k_softmax_v2',
                        'Pack reduction rows, losslessly widen to FP32, and use original V2 renorm=1 raw-score top-1 over <=2048 columns. Hierarchical merges gather original scores and original indices, never probabilities. Native index ordering must pass the unchanged complete original oracle; no approximate tie correction or external argmax. Argmin additionally invokes native constant -1 GEMM for half/BF16 or original FP32 routing weighted sum.')
    hashes = source_hashes(Path(__file__))
    if matrix is not None:
        directory = 'moe/moe_finalize_routing_v2' if full_minimum else 'gmm/grouped_matmul_add'
        hashes |= source_hashes(WORKSPACE / 'ascend_baselines/ops-transformer' / directory, matrix.vendor, Path(__file__).with_name('routing_arithmetic.py'))
    def cleanup():
        binding.cleanup()
        if matrix is not None:
            matrix.cleanup()
    return replace(result, cleanup=cleanup, source_sha256=result.source_sha256 | hashes)
