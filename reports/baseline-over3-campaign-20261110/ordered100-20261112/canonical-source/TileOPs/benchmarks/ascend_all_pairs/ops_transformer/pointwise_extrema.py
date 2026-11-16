"""Pointwise extrema through original attention raw maxima and native negation."""
from pathlib import Path


def pointwise_extrema(case):
    import torch
    from dataclasses import replace
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_transformer_graph import _attention
    from benchmarks.ascend_baselines import UnsupportedCase, WORKSPACE
    from .adapters import _candidate, _gemm
    from .hashing import source_hashes

    dtype = case.inputs[0].dtype
    if dtype not in (torch.float16, torch.bfloat16) or any(
            torch.is_tensor(t) and t.dtype != dtype for t in case.inputs):
        raise UnsupportedCase('Original attention extrema require matching FP16/BF16 inputs; no lossy narrowing.')
    op, p = case._call.signature.name, case.params
    for key in ('min', 'max', 'min_val', 'max_val'):
        bound = p.get(key)
        if isinstance(bound, (int, float)) and float(torch.tensor(bound, dtype=dtype).item()) != float(bound):
            raise UnsupportedCase('Scalar bound must be exactly representable in the original native input dtype.')
    binding = GraphCall('ops-transformer')

    def negate(x):
        weight = torch.full((1, 1), -1., dtype=x.dtype, device=x.device)
        out = _gemm(binding, x.reshape(-1, 1), weight, x.dtype).reshape(x.shape)
        binding.cleanup()
        return out

    def pair(a, b, minimum=False):
        a, b = torch.broadcast_tensors(a, b)
        packed = torch.stack((a, b), -1).reshape(-1, 2)
        chunks = []
        for start in range(0, packed.shape[0], 32768):
            rows = packed[start:start + 32768]
            keys = torch.nn.functional.pad(rows[:, :, None, None], (0, 127)).contiguous()
            query = torch.zeros((len(rows), 1, 1, 128), dtype=dtype, device=a.device)
            query[..., 0] = -1 if minimum else 1
            _, maximum = _attention(binding, query, keys, keys, 1.)
            chunks.append(maximum[:, 0, 0, 0].to(dtype))
            # Bound descriptor lifetimes for full 16M-element manifests. This
            # synchronization and cleanup remains inside the measured call.
            binding.cleanup()
        result = torch.cat(chunks).reshape(a.shape)
        return negate(result) if minimum else result

    def run(x, low=None, high=None):
        if op in ('MaximumFwdOp', 'MinimumFwdOp'):
            return pair(x, low, op == 'MinimumFwdOp')
        if op == 'AbsFwdOp':
            return pair(x, negate(x))
        if op == 'ReluFwdOp':
            return pair(x, torch.zeros((), dtype=dtype, device=x.device))
        if op in ('ClampScalarFwdOp', 'HardtanhFwdOp'):
            lo = p.get('min_val') if op == 'HardtanhFwdOp' else p.get('min')
            hi = p.get('max_val') if op == 'HardtanhFwdOp' else p.get('max')
            low = None if lo is None else torch.full((), lo, dtype=dtype, device=x.device)
            high = None if hi is None else torch.full((), hi, dtype=dtype, device=x.device)
        result = pair(x, low) if low is not None else x
        return pair(result, high, True) if high is not None else result

    uses_negation = op in ('AbsFwdOp', 'MinimumFwdOp', 'HardtanhFwdOp') or (
        op == 'ClampScalarFwdOp' and p.get('max') is not None) or (
        op == 'ClampTensorFwdOp' and len(case.inputs) > 2 and case.inputs[2] is not None)
    result = _candidate(binding, run, 'FlashAttentionScore' + ('+GroupedMatmulAdd' if uses_negation else ''),
                        'attention/flash_attention_score',
                        'Broadcast and pack two scalar keys; original attention raw maximum implements extrema. Signed basis queries and original constant -1 GEMM implement minima/absolute value. ReLU and clamps add exact constant/tensor bounds. All packing, native calls and bounded-lifetime synchronization are timed; no external floating comparisons, reductions or activation arithmetic.')
    return replace(result, source_sha256=result.source_sha256 | source_hashes(
        Path(__file__), WORKSPACE / 'ascend_baselines/ops-transformer/gmm/grouped_matmul_add'))
