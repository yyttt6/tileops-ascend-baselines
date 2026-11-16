"""Specialize the original SGL complete KDA API to zero-decay DeltaNet."""
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes,
)


def sgl_kda_deltanet(case):
    import torch
    from benchmarks.api import Implementation

    q, k, v = case.inputs[:3]
    if case.params.get("use_qk_l2norm_in_kernel", False):
        raise UnsupportedCase("The original SGL chunk KDA API has no Q/K normalization flag.")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise UnsupportedCase("The original SGL native chunk KDA API requires BF16 Q/K/V.")
    if q.shape != k.shape or any(d % 16 or not 16 <= d <= 256 for d in (q.shape[-1], v.shape[-1])):
        raise UnsupportedCase("Original KDA requires matching Q/K and aligned feature dimensions up to 256.")
    root = LIBRARIES / "sgl-kernel-npu"
    host = "python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk_kda_fwd.py"
    fn = local_function("sgl-kernel-npu", "python/sgl_kernel_npu",
                        "sgl_kernel_npu.fla.chunk_kda_fwd", "chunk_kda_fwd", host)

    scale = case.params.get("scale")
    scale = q.shape[-1] ** -0.5 if scale is None else scale

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        boundaries = offsets if offsets is not None else offsets_cpu
        if boundaries is not None:
            boundaries = boundaries.to(device=q.device, dtype=torch.int64).contiguous()
        gate = torch.zeros((*q.shape[:2], v.shape[2], q.shape[3]),
                           device=q.device, dtype=torch.float32)
        # The original API accepts [N,HV,K,V] FP32 states and returns both outputs.
        # Clone the private state so the manifest inputs remain protected.
        output = fn(q.contiguous(), k.contiguous(), v.contiguous(), gate,
                    beta.contiguous(), initial_state=None if initial is None else initial.clone().contiguous(),
                    cu_seqlens=boundaries, layout="BSND",
                    scale=scale,
                    use_gate_in_kernel=False, safe_gate=False,
                    state_v_first=False, output_final_state=True)
        return output[0], output[1]

    sources = [f for f in (root / "csrc/chunk_kda_fwd").rglob("*")
               if f.suffix in (".cpp", ".h", ".hpp")]
    return Candidate(
        Implementation(run), "sgl_kernel_npu.fla.chunk_kda_fwd(g=0)",
        "Original full KDA API with zero channel log-decay equals ungated DeltaNet; preserve beta, scale, FP32 state and sequence boundaries; no native stage or schedule changes.",
        source_hashes(Path(__file__), Path(__file__).with_name("ascend_sgl_delta_cases.py"),
                      root / host, root / "csrc/pytorch_extensions.cpp",
                      root / "python/sgl_kernel_npu/sgl_kernel_npu/__init__.py",
                      root / "python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so",
                      *sources),
        limitations=("Both manifest outputs and input preservation must pass original tolerances.",),
    )
