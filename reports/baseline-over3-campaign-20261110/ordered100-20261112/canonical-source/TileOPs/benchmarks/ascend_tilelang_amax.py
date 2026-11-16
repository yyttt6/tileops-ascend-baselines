"""Amax is the value output of the original generic TopK with k=1."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
from benchmarks.ascend_tilelang_generic import _original_jit


def tilelang_amax(case):
    from benchmarks.api import Implementation

    if case.params["dim"] != -1 or case.inputs[0].ndim != 2:
        raise UnsupportedCase("This pilot reduces each packed row.")
    fn, hashes = _original_jit("examples/topk/example_topk.py", "tl_topk")

    def run(x):
        values, _indices = fn(x.float().contiguous(), 1, 1, threads=128)
        out = values.to(x.dtype)
        return out if case.params["keepdim"] else out.squeeze(-1)

    return Candidate(
        Implementation(run), "tilelang::tl_topk(k=1).values",
        "Original TopK value output with k1, row tile1, FP16 exact widening and final cast; original reduction schedule.",
        source_hashes(Path(__file__), Path(__file__).with_name("ascend_tilelang_generic.py")) | hashes,
    )
