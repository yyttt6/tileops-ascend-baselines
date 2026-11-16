"""Small API/shape adapters for original rotation and quantization kernels."""

from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes,
)


def _rope_tables(torch, x, base):
    half = x.shape[-1] // 2
    inv = 1. / (base ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    angles = torch.outer(torch.arange(x.shape[0], device=x.device, dtype=torch.float32), inv)
    # Preserve the manifest's table rounding before calling FP32-only APIs.
    return angles.cos().to(x.dtype), angles.sin().to(x.dtype)


def tilekernels_rope(case):
    import torch
    from benchmarks.api import Implementation

    x, = case.inputs
    if x.ndim != 2 or x.shape[-1] not in (64, 128) or case.params["rope_layout"] != "neox":
        raise UnsupportedCase("The first sample is a fixed 1D NeoX rotation.")
    fn = local_function(
        "TileKernels", ".", "tile_kernels.transform.rope_kernel", "apply_rotary",
        "tile_kernels/transform/rope_kernel.py",
    )
    base = case.params.get("base", 10000.)

    def run(x):
        cos, sin = _rope_tables(torch, x, base)
        cache = torch.cat((cos, sin), dim=-1).float()
        # FP16 -> FP32 is exact. The original API modifies its input, so use a
        # private buffer. Both casts and table creation belong to every call.
        q = x.float().clone().view(x.shape[0], 1, x.shape[1])
        fn(q, cache, interleaved=False)
        return q.view_as(x).to(x.dtype)

    root = LIBRARIES / "TileKernels/tile_kernels"
    return Candidate(
        Implementation(run), "TileKernels::apply_rotary",
        "1D -> TND; exact FP16-to-FP32 input widening; manifest-rounded tables; output cast.",
        source_hashes(Path(__file__), root / "transform/rope_kernel.py",
                      root / "transform/rope_asc.py", root / "config.py"),
        limitations=("Native FMA rounding must pass the unchanged manifest verifier.",),
    )


def tilekernels_fp8(case):
    import torch
    from benchmarks.api import Implementation

    x, = case.inputs
    hidden = x.shape[-1]
    if hidden not in (16, 32, 64, 128) or hidden % 64:
        raise UnsupportedCase("This sample uses the original full 64-channel scale group.")
    fn = local_function(
        "TileKernels", ".", "tile_kernels.quant.per_token_cast_kernel", "per_token_cast",
        "tile_kernels/quant/per_token_cast_kernel.py",
    )
    from tile_kernels.config import get_amax_clamp_for_quant
    if get_amax_clamp_for_quant("e4m3", False) != 1e-4:
        raise RuntimeError("Quantization clamp differs from this manifest's 1e-4 contract.")

    def run(x):
        # Ascend requires at least 128 channels. Keep groups of 64 and append
        # an independent zero group; the first group's amax is unchanged.
        packed = torch.nn.functional.pad(x.float().reshape(-1, hidden),
                                          (0, max(128 - hidden, 0)))
        quant, scale = fn(packed, "e4m3", hidden)
        # The manifest returns scales first. Do not replace native quantization
        # or its scaling formula with a Torch implementation.
        return scale[:, :1].reshape(x.shape[:-1]), quant[:, :hidden].contiguous().reshape(x.shape)

    root = LIBRARIES / "TileKernels/tile_kernels"
    return Candidate(
        Implementation(run), "TileKernels::per_token_cast",
        "Flatten, exact FP16 widening, pad an independent zero group to native width128, crop and reorder outputs.",
        source_hashes(Path(__file__), root / "quant/per_token_cast_kernel.py",
                      root / "quant/per_token_cast_asc.py", root / "quant/common.py",
                      root / "config.py"),
        limitations=("Scale and quantized-code rounding require the manifest's full two-output check.",),
    )
