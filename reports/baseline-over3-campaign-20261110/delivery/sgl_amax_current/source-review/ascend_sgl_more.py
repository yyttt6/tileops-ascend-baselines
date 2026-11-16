"""Use existing SGL selection and rotary kernels with manifest layouts."""
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes,
)
from benchmarks.ascend_transform_baselines import _rope_tables


def sgl_rope(case):
    import torch
    from benchmarks.api import Implementation

    x, = case.inputs
    if x.ndim != 2 or case.params["rope_layout"] != "neox":
        raise UnsupportedCase("This pilot uses the 1D NeoX layout.")
    host = "python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_rope_qk_mqa.py"
    fn = local_function("sgl-kernel-npu", "python/sgl_kernel_npu",
                        "sgl_kernel_npu.norm.fused_rope_qk_mqa", "fused_rope_qk_mqa", host)
    base = case.params.get("base", 10000.)

    def run(x):
        cos, sin = _rope_tables(torch, x, base)
        cache = torch.cat((cos, sin), dim=-1)
        q = x.unsqueeze(1)
        output, _ = fn(q, q, cache, x.shape[-1], True)
        return output.squeeze(1)

    return Candidate(
        Implementation(run), "sgl_kernel_npu::fused_rope_qk_mqa",
        "Singleton head dimension, original Q/K API with shared input and select Q output; per-call frequency tables.",
        source_hashes(Path(__file__), Path(__file__).with_name("ascend_transform_baselines.py"),
                      LIBRARIES / "sgl-kernel-npu" / host),
    )


def sgl_amax(case):
    from benchmarks.api import Implementation

    x, = case.inputs
    if x.ndim != 2 or case.params.get("dim", -1) not in (-1, 1):
        raise UnsupportedCase("This pilot reduces a matrix's final dimension.")
    host = "python/sgl_kernel_npu/sgl_kernel_npu/sample/argmax_softmax_prob.py"
    fn = local_function("sgl-kernel-npu", "python/sgl_kernel_npu",
                        "sgl_kernel_npu.sample.argmax_softmax_prob", "argmax_softmax_prob_fused", host)
    keepdim = case.params.get("keepdim", False)

    def run(x):
        indices, _ = fn(x.contiguous())
        values = x.gather(1, indices.unsqueeze(1))
        return values if keepdim else values.squeeze(1)

    return Candidate(
        Implementation(run), "sgl_kernel_npu::argmax_softmax_prob_fused + gather",
        "Select original argmax index and gather input values; native softmax overhead remains part of the call.",
        source_hashes(Path(__file__), LIBRARIES / "sgl-kernel-npu" / host,
                      LIBRARIES / "sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/utils/triton_utils.py"),
        limitations=("Finite frozen-case inputs; the original kernel's nonfinite and signed-zero semantics are not generalized.",),
    )
