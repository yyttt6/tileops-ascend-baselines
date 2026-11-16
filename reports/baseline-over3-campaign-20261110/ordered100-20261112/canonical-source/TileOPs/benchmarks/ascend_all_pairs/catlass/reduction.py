"""Original CATLASS SIMT full-row sum, retaining its 128-thread schedule."""
from __future__ import annotations
from math import prod
import os
from pathlib import Path
from .build import CACHE, hashes
from .vector import DSL, _load_original


def sum_reduction(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    x = case.inputs[0]
    p = case.params
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase("The original SIMT sum uses FP32 accumulation; integer widening is not lossless")
    output_dtype = getattr(torch, p["dtype"]) if p.get("dtype") else x.dtype
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase("Original FP32 sum cannot implement FP64/int64 accumulation")
    axes = p.get("dim")
    axes = tuple(range(x.ndim)) if axes is None else ((axes,) if isinstance(axes, int) else tuple(axes))
    axes = tuple(a % x.ndim for a in axes)
    if len(set(axes)) != len(axes):
        raise ValueError("Repeated reduction dimension")
    rest = tuple(i for i in range(x.ndim) if i not in axes)
    width = prod(x.shape[i] for i in axes)
    rows = prod(x.shape[i] for i in rest)
    output_shape = (tuple(1 if i in axes else x.shape[i] for i in range(x.ndim))
                    if p.get("keepdim") else tuple(x.shape[i] for i in rest))
    source = DSL / "examples/end_to_end/simt/reduction_simt/reduction_simt.py"
    tla, module = _load_original("reduction_simt", source_file=source)
    # Shape specialization only: keep THREADS=128, shared-partial accumulation,
    # barrier, and thread-zero final fold exactly as in the original kernel.
    module.N_ELE = width
    sample_x = torch.empty(width, dtype=torch.float32, device=x.device)
    sample_out = torch.empty(1, dtype=torch.float32, device=x.device)
    def runtime(t):
        return tla.from_dlpack(t, layout_tag=tla.arch.RowMajor)
    old_cache = os.environ.get("CATLASS_DSL_CACHE_DIR")
    os.environ["CATLASS_DSL_CACHE_DIR"] = str(CACHE / "dsl")
    try:
        artifact = tla.compile(module.reduction_simt, runtime(sample_x), runtime(sample_out), options="--npu-arch 3510")
    finally:
        if old_cache is None:
            os.environ.pop("CATLASS_DSL_CACHE_DIR", None)
        else:
            os.environ["CATLASS_DSL_CACHE_DIR"] = old_cache
    files = [source, Path(__file__), Path(__file__).with_name("vector.py"), Path(artifact.kernel_binary_path).resolve()]
    files += list((DSL / "catlass").rglob("*.py")) + list((DSL / "catlass").rglob("*.so"))
    def run(x):
        matrix = x.permute(rest + axes).contiguous().reshape(rows, width).float()
        out = torch.empty(rows, dtype=torch.float32, device=x.device)
        for row in range(rows):
            artifact(runtime(matrix[row]), runtime(out[row:row + 1]), block_num=1,
                     stream=torch.npu.current_stream(x.device).npu_stream)
        return out.to(output_dtype).reshape(output_shape)
    return Candidate(Implementation(run), "CATLASS TLA DSL reduction_simt full-row FP32 sum",
                     "Permute/pack reduction axes each call; N_ELE specializes the input extent, all original 128-thread accumulation/synchronization retained; independent rows launch separately; exact input widening and output storage conversion.",
                     hashes(files), timing_mode="host_roundtrip", limitations=("DSL bindings missing in current snapshot; no numeric device receipt.",))
