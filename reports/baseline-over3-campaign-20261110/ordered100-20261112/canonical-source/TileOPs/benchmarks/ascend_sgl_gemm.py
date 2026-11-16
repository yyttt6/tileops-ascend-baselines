"""Single-adapter LoRA is an ordinary matrix product at unit scale."""
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, _owned_native, source_hashes,
)


def sgl_gemm(case):
    import importlib
    import sys
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    a, b = case.inputs
    if a.dtype != torch.bfloat16 or b.dtype != a.dtype:
        raise UnsupportedCase("This first LoRA host binding uses BF16 inputs.")
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
    # This own host entry launches EXEC_KERNEL_CMD(sgemmc_shrink); it is not a
    # stock aclnnMatmul wrapper or the optional CATLASS build target.
    _owned_native(binary, ["_ZN6sglang10npu_kernel13sgemmc_shrinkERN2at6TensorES3_S3_S3_S3_S3_S3_l"])
    indices = torch.zeros(m, dtype=torch.int32, device=a.device)
    lengths = torch.ones(m, dtype=torch.int32, device=a.device)
    ranks = torch.tensor([n], dtype=torch.int32, device=a.device)
    scales = torch.ones(1, dtype=torch.float16, device=a.device)
    # INNER_T is float, but Y_T is scalar_t: the final native store is BF16.
    output = torch.empty((m, n), dtype=a.dtype, device=a.device)
    padded_k = max(k, ((n // 16) + 1) * 16)

    def run(a, b):
        x = (a.T if ta else a).contiguous()
        weight = (b if tb else b.T).contiguous()
        # The native API requires hidden_in > hidden_out. Zero padding on K
        # preserves the product while satisfying that host shape constraint.
        x = F.pad(x, (0, padded_k - k))
        weight = F.pad(weight, (0, padded_k - k)).unsqueeze(0)
        torch.ops.npu.sgemmc_shrink(x, weight, indices, lengths, ranks, scales, output, 1)
        return output

    sources = [f for f in (root / "csrc/lora").rglob("*")
               if f.suffix in (".cpp", ".h", ".hpp")]
    return Candidate(
        Implementation(run), "sgl_kernel_npu::sgemmc_shrink (one LoRA, scale=1)",
        "B transpose, singleton LoRA, zero-pad K for hidden_in>hidden_out; native BF16 output; original schedule.",
        source_hashes(Path(__file__), binary, package / "__init__.py",
                      root / "csrc/pytorch_extensions.cpp", *sources),
        limitations=("This frozen shape hung in classics_006 and is quarantined; dtype correction is not a validated fix.",),
    )
