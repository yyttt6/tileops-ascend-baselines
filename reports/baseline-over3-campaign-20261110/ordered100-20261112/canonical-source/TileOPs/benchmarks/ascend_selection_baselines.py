"""Adapt original selection kernels through index/value layout operations."""

from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes


def tilekernels_amax(case):
    from benchmarks.api import Implementation

    x, = case.inputs
    dim = case.params.get("dim", -1)
    if x.ndim != 2 or dim not in (-1, 1):
        raise UnsupportedCase("This first sample reduces the final axis of a matrix.")
    fn = local_function(
        "TileKernels", ".", "tile_kernels.moe.topk_gate_kernel", "topk_gate",
        "tile_kernels/moe/topk_gate_kernel.py",
    )
    keepdim = case.params.get("keepdim", False)

    def run(x):
        # Exact widening allows the original FP32 argmax kernel to compare
        # the FP16 values. Gather preserves their original bits/dtype.
        indices = fn(x.float().contiguous(), 1)
        values = x.gather(1, indices)
        return values if keepdim else values.squeeze(1)

    root = LIBRARIES / "TileKernels/tile_kernels"
    return Candidate(
        Implementation(run), "TileKernels::topk_gate(k=1) + gather",
        "Exact FP16-to-FP32 widening; original native argmax selection; gather selected input value.",
        source_hashes(Path(__file__), root / "moe/topk_gate_kernel.py",
                      root / "moe/topk_gate_asc.py", root / "config.py"),
        limitations=("The original topk_gate rejects nonfinite score rows; this frozen manifest input is finite.",
                     "Equal maxima use the smallest index; general signed-zero semantics are not claimed."),
    )
