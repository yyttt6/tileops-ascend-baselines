"""Compile and call the original upstream Ascend examples with manifest shapes."""

from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, _rms_layout, local_function, source_hashes,
)


def _runtime():
    import tilelang

    expected = (LIBRARIES / "tilelang/tilelang/__init__.py").resolve()
    if Path(tilelang.__file__).resolve() != expected:
        # The local source was installed as a normal wheel. Verify every Python
        # source file before accepting it, then bind its actual native libraries
        # into the candidate receipt.
        installed = Path(tilelang.__file__).resolve().parent
        for path in expected.parent.rglob("*.py"):
            copy = installed / path.relative_to(expected.parent)
            if not copy.is_file() or path.read_bytes() != copy.read_bytes():
                raise RuntimeError(f"Installed tilelang differs from selected source: {path}")
    return tilelang


def _runtime_hashes(tilelang):
    root = Path(tilelang.__file__).resolve().parent
    return source_hashes(*root.rglob("*.py"), *list((root / "lib").glob("*.so")))


def tilelang_gemm(case):
    import torch
    from benchmarks.api import Implementation

    tilelang = _runtime()
    a, b = case.inputs
    ta, tb = case.params["trans_a"], case.params["trans_b"]
    m, k = a.T.shape if ta else a.shape
    kb, n = b.T.shape if tb else b.shape
    if k != kb or a.dtype != torch.bfloat16 or b.dtype != a.dtype:
        raise UnsupportedCase("This pilot selects the original BF16 GEMM configuration.")
    if any(v % 256 for v in (m, n, k)):
        raise UnsupportedCase("This first adapter uses the original complete 256-element tiles.")
    host = "examples/ascend/example_gemm.py"
    builder = local_function("tilelang", "examples/ascend", "example_gemm", "gemm", host)
    kernel = tilelang.compile(builder(m, k, n, dtype="bfloat16", out_dtype="bfloat16"),
                              target="ascend", out_idx=-1)

    def run(a, b):
        # Original example computes X @ W.T. Materialize its required W layout
        # on each call; no input-dependent copy is cached in the factory.
        x = (a.T if ta else a).contiguous()
        w = (b if tb else b.T).contiguous()
        return kernel(x, w)

    return Candidate(
        Implementation(run), "tilelang.examples.ascend.example_gemm.gemm",
        "Original BF16 persistent GEMM; manifest sizes and B transpose only; original schedule defaults.",
        source_hashes(Path(__file__), LIBRARIES / "tilelang" / host) | _runtime_hashes(tilelang),
    )


def tilelang_rms(case):
    from benchmarks.api import Implementation

    tilelang = _runtime()
    x, shape, hidden, eps, pack = _rms_layout(case)
    rows = x.numel() // hidden
    if rows % 64:
        raise UnsupportedCase("This first adapter requires the original complete row tiles.")
    host = "examples/ascend/example_rmsnorm.py"
    builder = local_function("tilelang", "examples/ascend", "example_rmsnorm", "rms_norm_fwd", host)
    # The original example's DMA buffers are FP32 regardless of dtype. Widen
    # the BF16 inputs exactly at the host boundary, retaining its FP32 schedule.
    kernel = tilelang.compile(builder(rows, hidden, dtype="float32"),
                              target="ascend", out_idx=[1, 3])

    def run(x, weight):
        x2, gamma = pack(x, weight)
        output, _rstd = kernel(x2.float().flatten(), gamma.float(), eps)
        return output.view(shape).to(x.dtype)

    return Candidate(
        Implementation(run), "tilelang.examples.ascend.example_rmsnorm.rms_norm_fwd",
        "Original FP32 example; exact BF16 widening on each call, flatten, select output and cast to BF16.",
        source_hashes(Path(__file__), LIBRARIES / "tilelang" / host) | _runtime_hashes(tilelang),
    )
