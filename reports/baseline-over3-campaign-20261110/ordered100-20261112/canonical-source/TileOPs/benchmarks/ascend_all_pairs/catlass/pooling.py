"""Independent pooling windows packed into the original 64-lane max entry.

No window is split across reductions and no partial result is reduced again.
The copied DSL has no generated bindings; these complete adapters stay deferred.
"""
from __future__ import annotations

from itertools import product
from math import prod
import os
from pathlib import Path

from .build import CACHE, hashes
from .convolution import tuple_param
from .vector import EXAMPLES, DSL, _load_original


def geometry(shape, kernel, stride, padding, dilation, ceil_mode):
    rank = len(shape)
    kernel = tuple_param(kernel, rank)
    stride = tuple_param(stride if stride is not None else kernel, rank)
    padding, dilation = tuple_param(padding, rank), tuple_param(dilation, rank)
    effective = tuple(d * (k - 1) + 1 for d, k in zip(dilation, kernel))
    output, pads = [], []
    for size, e, step, pad in zip(shape, effective, stride, padding):
        extent = (size + 2 * pad - e + (step - 1 if ceil_mode else 0)) // step + 1
        if ceil_mode and (extent - 1) * step >= size + pad:
            extent -= 1
        output.append(extent)
        pads.append((pad, max(pad, (extent - 1) * step + e - size - pad)))
    if min(output) <= 0:
        raise ValueError("Nonpositive pooling output dimension")
    return kernel, stride, dilation, effective, tuple(pads), tuple(output)


def adaptive_windows(shape, output):
    result = []
    for index in product(*(range(n) for n in output)):
        result.append(tuple((i * size // n, ((i + 1) * size + n - 1) // n)
                            for i, size, n in zip(index, shape, output)))
    return result


class Max64Engine:
    def __init__(self, device):
        import torch
        tla, module = _load_original("reduction_ops")
        module._set_op("max")
        self.tla = tla
        sample_x = torch.empty(128, dtype=torch.float32, device=device)
        sample_z = torch.empty(2, dtype=torch.float32, device=device)
        old_cache = os.environ.get("CATLASS_DSL_CACHE_DIR")
        os.environ["CATLASS_DSL_CACHE_DIR"] = str(CACHE / "dsl")
        try:
            self.artifact = tla.compile(module.reduction_op, self.runtime(sample_x), self.runtime(sample_z),
                                        options="--npu-arch 3510")
        finally:
            if old_cache is None:
                os.environ.pop("CATLASS_DSL_CACHE_DIR", None)
            else:
                os.environ["CATLASS_DSL_CACHE_DIR"] = old_cache
        files = [Path(__file__), Path(__file__).with_name("vector.py"),
                 Path(__file__).with_name("convolution.py"), EXAMPLES / "reduction_ops.py",
                 EXAMPLES / "vector_op_harness.py", Path(self.artifact.kernel_binary_path).resolve()]
        files += list((DSL / "catlass").rglob("*.py")) + list((DSL / "catlass").rglob("*.so"))
        self.source_sha256 = hashes(files)

    def runtime(self, tensor):
        return self.tla.from_dlpack(tensor, layout_tag=self.tla.arch.RowMajor)

    def __call__(self, matrix):
        import torch
        rows, width = matrix.shape
        if not 0 < width <= 64:
            raise ValueError("Original max entry requires a whole window of at most64 elements")
        native_rows = (rows + 1) // 2 * 2
        # Each two contiguous64-element windows match the original128->2 API.
        # Pack bounded batches to avoid an expanded64-lane copy of the full image.
        output = torch.empty(native_rows, dtype=torch.float32, device=matrix.device)
        for base in range(0, native_rows, 2048):
            stop = min(base + 2048, native_rows)
            valid = min(stop, rows) - base
            packed = torch.full((stop - base, 64), -float("inf"), dtype=torch.float32, device=matrix.device)
            packed[:valid, :width].copy_(matrix[base:base + valid])
            for row in range(0, stop - base, 2):
                self.artifact(self.runtime(packed[row:row + 2].flatten()),
                              self.runtime(output[base + row:base + row + 2]), block_num=1,
                              stream=torch.npu.current_stream(matrix.device).npu_stream)
        return output[:rows].to(matrix.dtype)


def max_pool(case):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    x = case.inputs[0]
    p = case.params
    rank = len(p["output"][0]) - 2
    if rank not in (1, 2, 3):
        raise UnsupportedCase("Expected batched pooling spatial rank1/2/3")
    kernel, stride, dilation, effective, pads, output_shape = geometry(
        x.shape[2:], p["kernel_size"], p.get("stride"), p.get("padding", 0),
        p.get("dilation", 1), p.get("ceil_mode", False))
    if prod(kernel) > 64:
        raise UnsupportedCase("Complete pooling window exceeds original64-lane reduction")
    engine = Max64Engine(x.device)

    def run(x):
        import torch.nn.functional as F
        value = F.pad(x, tuple(v for pair in reversed(pads) for v in pair), value=-float("inf"))
        for axis in range(rank):
            value = value.unfold(2 + axis, effective[axis], stride[axis])
        selection = [slice(None)] * (2 + rank) + [slice(None, None, d) for d in dilation]
        value = value[tuple(selection)].contiguous().reshape(-1, prod(kernel))
        return engine(value).reshape(*x.shape[:2], *output_shape)

    return Candidate(Implementation(run), "CATLASS TLA DSL reduction_ops.reduction_op / original max option",
                     "Per-run spatial negative-infinity padding, dilated/strided window view, and complete window packing into64 lanes; original128->2 max entry per independent pair; crop and restore output. No split reductions or index emulation.",
                     engine.source_sha256, timing_mode="host_roundtrip", limitations=("Missing DSL bindings; no numerical receipt.",
                                                        "Millions of independent window launches may be prohibitively slow; all costs must be timed.",))


def adaptive_max_pool(case):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    x = case.inputs[0]
    size = tuple(case.params["output_size"])
    windows = adaptive_windows(x.shape[-2:], size)
    if any(prod(hi - lo for lo, hi in window) > 64 for window in windows):
        raise UnsupportedCase("An adaptive window exceeds the original64-lane max reduction")
    engine = Max64Engine(x.device)

    def run(x):
        import torch
        output = torch.empty((*x.shape[:-2], *size), dtype=x.dtype, device=x.device)
        for index, window in zip(product(*(range(n) for n in size)), windows, strict=True):
            slices = tuple(slice(lo, hi) for lo, hi in window)
            rows = x[(..., *slices)].contiguous().reshape(-1, prod(hi - lo for lo, hi in window))
            output[(..., *index)] = engine(rows).reshape(x.shape[:-2])
        return output

    return Candidate(Implementation(run), "CATLASS TLA DSL reduction_ops.reduction_op / original max option",
                     "Exact floor/ceil adaptive-window slicing, lossless FP32 packing and negative-infinity padding into original64-lane max entry; restore all output windows, no reduction splitting or added math.",
                     engine.source_sha256, timing_mode="host_roundtrip", limitations=("Missing DSL bindings; no numerical receipt.",))
