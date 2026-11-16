"""Inclusive prefix sums as original native products with triangular constants."""
from pathlib import Path


def prefix_sum(case):
    import torch
    from dataclasses import replace
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    from .adapters import _candidate
    from .hashing import source_hashes
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16):
        raise UnsupportedCase('Original GroupedMatmulAdd inputs are FP16/BF16; FP32 or integer narrowing is forbidden.')
    binding = GraphCall('ops-transformer')
    dim = case.params['dim']

    def run(x):
        packed = x.movedim(dim, -1).contiguous()
        width = packed.shape[-1]
        rows = packed.reshape(-1, width)
        carry = torch.zeros((len(rows), 1), dtype=torch.float32, device=x.device)
        parts = []
        for start in range(0, width, 128):
            block = rows[:, start:start + 128].contiguous()
            count = block.shape[1]
            triangular = torch.ones((count, count), dtype=x.dtype, device=x.device).triu()
            out = carry.expand(len(rows), count).clone()[None]
            group = torch.full((1,), count, dtype=torch.int64, device=x.device)
            binding.call('GroupedMatmulAdd', [block.T.contiguous(), triangular, group, out], [out],
                         {'transpose_x': True, 'transpose_weight': False,
                          'group_type': 2, 'group_list_type': 0})
            carry = out[0, :, -1:].clone()
            parts.append(out[0].to(x.dtype))
            binding.cleanup()
        return torch.cat(parts, -1).reshape(packed.shape).movedim(-1, dim).contiguous()

    result = _candidate(binding, run, 'GroupedMatmulAdd', 'gmm/grouped_matmul_add',
                        'Move scan axis into native GEMM K; exact upper-triangular zero/one constants produce each inclusive prefix. Carry original FP32 final output as the next native accumulator before output casting. Complete axis packing, copies, native calls and synchronization are timed; no Torch cumulative arithmetic.')
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__)))
