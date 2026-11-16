"""Layout/API adapters only; all arithmetic belongs to original TileKernels."""
from pathlib import Path
from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes


def silu_and_mul(case):
    import torch
    from benchmarks.api import Implementation
    fn = local_function("TileKernels", ".", "tile_kernels.quant.swiglu_forward_kernel",
                        "swiglu_forward", "tile_kernels/quant/swiglu_forward_kernel.py")
    dtype = case.inputs[0].dtype
    def run(x):
        # FP16 values widen losslessly; native FP32 output is restored to contract dtype.
        half = x.shape[-1] // 2
        # SwiGLU is pointwise: pair-preserving reshape avoids the original full-row UB limit.
        left = x[..., :half].float().contiguous().reshape(-1, 1024)
        right = x[..., half:].float().contiguous().reshape(-1, 1024)
        paired = torch.cat((left, right), dim=-1)
        return fn(paired, "fp32").to(dtype).reshape(*x.shape[:-1], half)
    root = LIBRARIES / "TileKernels/tile_kernels/quant"
    return Candidate(Implementation(run), "tile_kernels.quant.swiglu_forward_kernel.swiglu_forward",
                     "Pair-preserving 1024-column reshape; exact FP16/BF16 widening; native FP32 SwiGLU; restore output dtype on every call.",
                     source_hashes(Path(__file__), root / "swiglu_forward_kernel.py", root / "swiglu_forward_asc.py", root / "common.py", root.parent / "config.py"))
