"""Convolution as operand packing and original CATLASS full-reduction GEMM.

No arithmetic is performed by torch: pad/unfold/stride/permute/copy implement
the im2col layout. The reduction and optional bias are CATLASS's own template.
"""
from __future__ import annotations

from math import prod
from pathlib import Path
from .build import hashes
from .gemm import GemmEngine, _candidate


def tuple_param(value, rank):
    result = (value,) * rank if isinstance(value, int) else tuple(value)
    if len(result) != rank:
        raise ValueError(f"Expected {rank} spatial parameters")
    return result


def geometry(shape, kernel, stride, padding, dilation):
    rank = len(kernel)
    stride, dilation = tuple_param(stride, rank), tuple_param(dilation, rank)
    effective = tuple(d * (k - 1) + 1 for d, k in zip(dilation, kernel))
    if padding == "same":
        if any(s != 1 for s in stride):
            raise ValueError("The workload same-padding contract requires stride one")
        pads = tuple(((e - 1) // 2, e - 1 - (e - 1) // 2) for e in effective)
    elif padding == "valid":
        pads = ((0, 0),) * rank
    else:
        pads = tuple((p, p) for p in tuple_param(padding, rank))
    output = tuple((s + lo + hi - e) // t + 1
                   for s, (lo, hi), e, t in zip(shape, pads, effective, stride))
    if min(output) <= 0:
        raise ValueError("Nonpositive convolution output dimension")
    return stride, dilation, effective, pads, output


def convolution(case):
    from dataclasses import replace
    x, weight, *extras = case.inputs
    p = case.params
    rank = x.ndim - 2
    if rank not in (1, 2, 3) or weight.ndim != x.ndim:
        raise ValueError("Expected N,C,(D),H,W input and O,I,... filter")
    groups = p.get("groups", 1)
    cin, cout = x.shape[1] // groups, weight.shape[0] // groups
    if x.shape[1] % groups or weight.shape[0] % groups or weight.shape[1] != cin:
        raise ValueError("Invalid convolution channel groups")
    stride, dilation, effective, pads, output_shape = geometry(
        x.shape[2:], weight.shape[2:], p.get("stride", 1), p.get("padding", 0), p.get("dilation", 1))
    has_bias = bool(extras and extras[0] is not None)
    engine = GemmEngine(x.dtype, bias=has_bias)

    def run(x, weight, bias=None):
        import torch
        import torch.nn.functional as F
        if (bias is not None) != has_bias:
            raise ValueError("Bias presence changed after candidate construction")
        # Spatial padding is data movement only; no torch convolution/matmul.
        x = F.pad(x, tuple(v for pair in reversed(pads) for v in pair))
        batch = x.shape[0]
        result = torch.empty((batch, weight.shape[0], *output_shape), dtype=x.dtype, device=x.device)
        for group in range(groups):
            patch = x[:, group * cin:(group + 1) * cin]
            for axis in range(rank):
                patch = patch.unfold(2 + axis, effective[axis], stride[axis])
            select = [slice(None)] * (2 + rank) + [slice(None, None, d) for d in dilation]
            patch = patch[tuple(select)]
            axes = [0, *range(2, 2 + rank), 1, *range(2 + rank, 2 + 2 * rank)]
            lhs = patch.permute(axes).contiguous().reshape(batch * prod(output_shape), -1)
            rhs = weight[group * cout:(group + 1) * cout].reshape(cout, -1).T
            b = bias[group * cout:(group + 1) * cout] if bias is not None else None
            value = engine(lhs, rhs, b).reshape(batch, *output_shape, cout)
            value = value.permute(0, rank + 1, *range(1, rank + 1))
            result[:, group * cout:(group + 1) * cout].copy_(value)
        return result

    candidate = _candidate(run, engine,
        "Per-call spatial pad + unfold/strided filter view + im2col packing; each independent channel group uses one full native GEMM reduction and original bias option; restore N,C,spatial output. No reduction splitting or kernel/schedule edits.")
    return replace(candidate, source_sha256=candidate.source_sha256 | hashes([Path(__file__)]),
                   limitations=candidate.limitations + ("Packing may consume substantial temporary memory; all packing and native calls must be timed.",))
