"""Shape/layout bindings for additional unmodified upstream Ascend examples."""
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, _add_layout, local_function, source_hashes,
)
from benchmarks.ascend_tilelang_baselines import _runtime, _runtime_hashes


def tilelang_add(case):
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    tilelang = _runtime()
    x, _, shape, rows, hidden, pack = _add_layout(case)
    size = rows * hidden
    native_size = ((size + 8192 * 64 - 1) // (8192 * 64)) * (8192 * 64)
    host = "examples/ascend/example_vecadd.py"
    builder = local_function("tilelang", "examples/ascend", "example_vecadd", "vector_add", host)
    kernel = tilelang.compile(builder(native_size, mode="simt"), target="ascend", out_idx=-1)

    def run(x, y):
        # The native API uses FP32. Widen BF16 exactly, then round the sum once
        # back to the frozen output dtype; all casts belong to this invocation.
        a = F.pad(pack(x).flatten().float(), (0, native_size - size))
        b = F.pad(pack(y).flatten().float(), (0, native_size - size))
        return kernel(a, b)[:size].to(x.dtype).view(shape)

    return Candidate(
        Implementation(run), "tilelang::example_vecadd.vector_add(simt)",
        "Exact FP16/BF16 widening, broadcast and independent tail padding to complete native tiles; crop/cast output; original pipeline unchanged.",
        source_hashes(Path(__file__), LIBRARIES / "tilelang" / host) | _runtime_hashes(tilelang),
    )


def tilelang_fp8(case):
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    tilelang = _runtime()
    x, = case.inputs
    if x.shape[-1] != 64 or x.numel() // 64 % 32:
        raise UnsupportedCase("This sample pads independent 64-value groups to the original 128-value API.")
    host = "examples/ascend/example_per_token_cast_to_fp8.py"
    builder = local_function(
        "tilelang", "examples/ascend", "example_per_token_cast_to_fp8",
        "per_token_cast_to_fp8", host)
    kernel = builder(x.numel() // 64, 128, mode="simt")
    shape = tuple(x.shape)

    def run(x):
        # Padding each group with zero preserves absmax and the original clamp.
        padded = F.pad(x.float().reshape(-1, 64), (0, 64))
        quant, scale = kernel(padded)
        return scale.reshape(shape[:-1]), quant[:, :64].contiguous().reshape(shape)

    return Candidate(
        Implementation(run), "tilelang::per_token_cast_to_fp8(simt)",
        "Exact FP16 widening; per-group zero pad64->128 and output crop; original quantization/scale and schedule.",
        source_hashes(Path(__file__), LIBRARIES / "tilelang" / host) | _runtime_hashes(tilelang),
    )


def tilelang_gqa(case):
    """Use the unchanged BF16 implementation; FP16 source rewriting is retired."""
    from benchmarks.ascend_tilelang_attention_cases import tilelang_gqa_cases
    return tilelang_gqa_cases(case)
