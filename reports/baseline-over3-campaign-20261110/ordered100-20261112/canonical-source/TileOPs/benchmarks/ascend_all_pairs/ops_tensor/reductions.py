"""Constant-operand selections of original GEMM, with no external reduction math.

Sum is X @ ones. Mean is X @ full(1/R). All accumulator work is performed by
the unchanged vendor FP32 GEMM; input widening is exact and output dtype is restored.
"""
from itertools import product
from math import prod

from .adapters import _candidate, _check, _float_inputs, _gemm_engine
from .contract import pooling_plan


def _constant_contraction(mm, rows, coefficient):
    import torch
    weights = torch.full((rows.shape[1], 1), coefficient, device=rows.device, dtype=torch.float32)
    return mm(rows.float(), weights)


def sum_reduction(case):
    return _reduction(case, "SumFwdOp", mean=False)


def mean_reduction(case):
    return _reduction(case, "MeanFwdOp", mean=True)


def _reduction(case, op, mean):
    _check(op, case)
    import torch
    mm, hashes = _gemm_engine()
    dims, keepdim = case.params.get("dim"), case.params.get("keepdim", False)
    output_dtype = getattr(torch, case.params["output"][1])

    def run(x):
        _float_inputs(x)
        axes = tuple(range(x.ndim)) if dims is None else ((dims,) if isinstance(dims, int) else tuple(dims))
        if not axes:
            axes = tuple(range(x.ndim))
        axes = tuple(d % x.ndim for d in axes) if x.ndim else ()
        kept = tuple(i for i in range(x.ndim) if i not in axes)
        reduction = prod(x.shape[d] for d in axes)
        rows = x.permute(kept + axes).reshape(-1, reduction)
        result = _constant_contraction(mm, rows, 1.0 / reduction if mean else 1.0)
        shape = tuple(1 if i in axes else x.shape[i] for i in range(x.ndim)) if keepdim else tuple(x.shape[i] for i in kept)
        return result.to(output_dtype).reshape(shape)

    return _candidate(run, "ops-tensor::LaunchKernel<float> / Blaze::GemmUniversal",
                      "Permute reduction axes and exactly widen to FP32; select original full GEMM with shape-only constant coefficients (ones for sum, 1/R for mean); crop and restore declared output dtype/keepdim. No new reduction tree or external arithmetic.", hashes)


def avg_pool(case):
    if "output_size" in case.params:
        op = "AdaptiveAvgPool2dFwdOp"
    else:
        kernel = case.params["kernel_size"]
        rank = len(kernel) if isinstance(kernel, (tuple, list)) else 1
        op = f"AvgPool{rank}dFwdOp"
    _check(op, case)
    import torch
    import torch.nn.functional as F
    plan = pooling_plan(op, case.params)
    mm, hashes = _gemm_engine()

    def run(x):
        _float_inputs(x)
        if plan["adaptive"]:
            output = torch.empty((*x.shape[:-2], *plan["output"]), dtype=x.dtype, device=x.device)
            for i, j, hs, he, ws, we in plan["windows"]:
                count = (he - hs) * (we - ws)
                rows = x[..., hs:he, ws:we].reshape(-1, count)
                result = _constant_contraction(mm, rows, 1.0 / count)
                output[..., i, j].copy_(result.to(x.dtype).reshape(x.shape[:-2]))
            return output
        rank = plan["rank"]
        lead = x.ndim - rank
        padding = tuple(v for pair in reversed(tuple(zip(plan["padding"], plan["right"]))) for v in pair)
        windows = F.pad(x, padding)
        for axis, (k, s) in enumerate(zip(plan["kernel"], plan["stride"])):
            windows = windows.unfold(lead + axis, k, s)
        output = torch.empty((*x.shape[:-rank], *plan["output"]), dtype=x.dtype, device=x.device)
        for coordinate in product(*(range(size) for size in plan["output"])):
            divisor = case.params.get("divisor_override")
            if divisor is None:
                counts = []
                for i, size, k, s, p in zip(coordinate, x.shape[-rank:], plan["kernel"], plan["stride"], plan["padding"]):
                    begin, end = i * s - p, i * s - p + k
                    low, high = (-p, size + p) if case.params.get("count_include_pad", True) else (0, size)
                    counts.append(max(0, min(end, high) - max(begin, low)))
                divisor = prod(counts)
            if divisor <= 0:
                raise ValueError("Pooling divisor must be positive")
            index = (slice(None),) * lead + coordinate
            rows = windows[index].reshape(-1, prod(plan["kernel"]))
            result = _constant_contraction(mm, rows, 1.0 / divisor)
            output[index].copy_(result.to(x.dtype).reshape(x.shape[:-rank]))
        return output

    return _candidate(run, "ops-tensor::LaunchKernel<float> / Blaze::GemmUniversal (constant pooling coefficients)",
                      "Per-call spatial/adaptive window packing and FP32 native contraction with constant reciprocal-divisor weights; exact operand widening, original ceil/include-pad/override divisor semantics, dtype restoration. All arithmetic is original GEMM.", hashes)


def mean_pooling(case):
    _check("MeanPoolingFwdOp", case)
    import torch
    mm, hashes = _gemm_engine()
    chunk = case.params["chunk_size"]

    def run(x, offsets=None, indices=None):
        _float_inputs(x)
        if offsets is None:
            intervals = [(start, min(start + chunk, x.shape[1])) for start in range(0, x.shape[1], chunk)]
        else:
            if indices is None:
                raise ValueError("Ragged mean pooling requires current indices")
            bounds = offsets.tolist()
            intervals = []
            for sequence, part in indices.tolist():
                if sequence < 0 or sequence + 1 >= len(bounds) or part < 0:
                    raise ValueError("Invalid ragged chunk metadata")
                begin = bounds[sequence] + part * chunk
                end = min(begin + chunk, bounds[sequence + 1])
                if begin < 0 or end > x.shape[1] or end <= begin:
                    raise ValueError("Ragged chunk has no valid extent")
                intervals.append((begin, end))
        output = torch.empty((x.shape[0], len(intervals), *x.shape[2:]), device=x.device, dtype=x.dtype)
        for index, (begin, end) in enumerate(intervals):
            rows = x[:, begin:end].movedim(1, -1).reshape(-1, end - begin)
            result = _constant_contraction(mm, rows, 1.0 / (end - begin))
            output[:, index].copy_(result.to(x.dtype).reshape(x.shape[0], *x.shape[2:]))
        return output

    return _candidate(run, "ops-tensor::LaunchKernel<float> / Blaze::GemmUniversal (sequence means)",
                      "Read current ragged offsets/indices inside run; pack each complete named chunk; original FP32 GEMM with constant 1/valid_length coefficients; retain chunk order and complete output. No cached metadata or external reduction.", hashes)
