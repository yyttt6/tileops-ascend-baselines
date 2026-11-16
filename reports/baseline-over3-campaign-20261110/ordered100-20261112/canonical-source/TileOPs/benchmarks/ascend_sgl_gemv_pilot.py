"""Focused alternative: original vector SGEMMV, distinct from quarantined SGEMMC."""
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, _owned_native, source_hashes,
)


def sgl_gemv_pilot(case):
    import importlib
    import sys
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    a, b = case.inputs
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise UnsupportedCase("Original vector LoRA accepts FP16/BF16 inputs.")
    ta, tb = case.params["trans_a"], case.params["trans_b"]
    m, k = a.T.shape if ta else a.shape
    kb, n = b.T.shape if tb else b.shape
    if kb != k:
        raise UnsupportedCase("GEMM dimensions disagree.")
    root = LIBRARIES / "sgl-kernel-npu"
    package = root / "python/sgl_kernel_npu/sgl_kernel_npu"
    sys.path.insert(0, str(package.parent))
    loaded = importlib.import_module("sgl_kernel_npu")
    if Path(loaded.__file__).resolve() != (package / "__init__.py").resolve():
        raise RuntimeError("SGL package resolved outside the inspected source.")
    binary = package / "lib/libsgl_kernel_npu.so"
    # This own host entry launches EXEC_KERNEL_CMD(sgemmv_shrink); it is not a
    # stock aclnnMatmul wrapper or the optional CATLASS build target.
    _owned_native(binary, ["_ZN6sglang10npu_kernel13sgemmv_shrinkERN2at6TensorES3_S3_S3_S3_S3_S3_l"])
    indices = torch.zeros(1, dtype=torch.int32, device=a.device)
    lengths = torch.tensor([m], dtype=torch.int32, device=a.device)
    ranks = torch.tensor([n], dtype=torch.int32, device=a.device)
    scales = torch.ones(1, dtype=torch.float16, device=a.device)
    if n > 1024:
        raise UnsupportedCase("This focused alternative is limited to the frozen 1024-column pilot until validated.")
    # SGEMMV, unlike SGEMMC, explicitly declares Y_T = float.
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    # Original DataCopy uses 32-byte blocks for FP16/BF16 operands.
    # Keep row starts and the complete reduction tail aligned after bias packing.
    padded_k = max(((k + 15) // 16) * 16, ((n // 16) + 1) * 16)

    def run(a, b):
        x = (a.T if ta else a).contiguous()
        weight = (b if tb else b.T).contiguous()
        # The native API requires hidden_in > hidden_out. Zero padding on K
        # preserves the product while satisfying that host shape constraint.
        x = F.pad(x, (0, padded_k - k))
        weight = F.pad(weight, (0, padded_k - k)).unsqueeze(0)
        torch.ops.npu.sgemmv_shrink(x, weight, indices, lengths, ranks, scales, output, 1)
        return output.to(a.dtype)

    sources = [f for f in (root / "csrc/lora").rglob("*")
               if f.suffix in (".cpp", ".h", ".hpp")]
    return Candidate(
        Implementation(run), "sgl_kernel_npu::sgemmv_shrink (one LoRA, scale=1)",
        "B transpose, singleton LoRA, zero-pad K for hidden_in>hidden_out; original FP32 vector-kernel output cast to public dtype; original schedule.",
        source_hashes(Path(__file__), binary, package / "__init__.py",
                      root / "csrc/pytorch_extensions.cpp", *sources),
        limitations=("Distinct original SGEMMV vector entry; quarantined SGEMMC is never called. Frozen BF16/FP16 shape must pass before expansion.",),
    )
