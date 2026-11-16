"""Adapters for the separate tilelang-ascend source tree and runtime."""

import ast
from pathlib import Path
import sys
import types

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, _add_layout, local_function, source_hashes,
)


def _runtime():
    import tilelang

    expected = (LIBRARIES / "tilelang-ascend/tilelang/__init__.py").resolve()
    if Path(tilelang.__file__).resolve() != expected:
        raise RuntimeError("Use baseline-run tilelang-ascend for its own compiler/runtime.")
    return tilelang


def _example_definitions(relative, name):
    """Load original imports/function definitions, excluding CLI/demo side effects.

    Some examples unconditionally clear the shared compiler cache, parse argv
    and run their demo at import. Keep their function ASTs byte-for-byte in
    structure and original line locations; do not rewrite any kernel body.
    """
    _runtime()
    path = LIBRARIES / "tilelang-ascend" / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
                 or (isinstance(node, ast.Assign) and
                     all(isinstance(t, ast.Name) and t.id == "pass_configs" for t in node.targets))]
    module_name = "_classic_tla_" + path.stem
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return getattr(module, name)


def tla_add(case):
    from benchmarks.api import Implementation

    x, _, shape, rows, hidden, pack = _add_layout(case)
    host = "examples/elementwise/elementwise_add.py"
    builder = _example_definitions(host, "vec_add")
    builder = _runtime().jit(target="pto", out_idx=[-1])(builder.__wrapped__)
    if rows % 128 or hidden % 256:
        # Pure addition is independent across elements. Repack complete tiles
        # without changing the original kernel's launch or vector schedule.
        total = rows * hidden
        if total % (128 * 256):
            raise UnsupportedCase("Repacked Add still requires complete128x256 tiles.")
        rows, hidden = total // 256, 256
    kernel = builder(rows, hidden, 128, 256, dtype=str(x.dtype).split(".")[-1])

    def run(x, y):
        return kernel(pack(x).reshape(rows, hidden), pack(y).reshape(rows, hidden)).view(shape)

    return Candidate(
        Implementation(run), "tilelang-ascend::elementwise_add.vec_add",
        "Original dtype/shape and 128x256 tile; select existing PTO backend; CLI/demo code omitted.",
        source_hashes(Path(__file__), LIBRARIES / "tilelang-ascend" / host),
    )


def tla_gemm(case):
    from benchmarks.api import Implementation

    a, b = case.inputs
    ta, tb = case.params["trans_a"], case.params["trans_b"]
    m, k = a.T.shape if ta else a.shape
    kb, n = b.T.shape if tb else b.shape
    if kb != k or m % 128 or n % 256 or k % 64:
        raise UnsupportedCase("This first adapter uses the original complete GEMM tiles.")
    host = "examples/gemm/example_gemm_pto_developer.py"
    builder = _example_definitions(host, "matmul")
    kernel = builder(m, n, k, 128, 256, 64, dtype=str(a.dtype).split(".")[-1])

    def run(a, b):
        return kernel((a.T if ta else a).contiguous(),
                      (b.T if tb else b).contiguous())

    return Candidate(
        Implementation(run), "tilelang-ascend::example_gemm_pto_developer.matmul",
        "Original PTO example, BF16 template parameters, original tile sizes, and input transpose/contiguous views.",
        source_hashes(Path(__file__), LIBRARIES / "tilelang-ascend" / host),
    )


def tla_rope(case):
    import torch
    from benchmarks.api import Implementation

    _runtime()
    x, *extra = case.inputs
    if extra or x.ndim != 2 or case.params["rope_layout"] != "neox":
        raise UnsupportedCase("This pilot specializes the fixed 1D NeoX case.")
    host = "examples/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py"
    fn = local_function(
        "tilelang-ascend", "examples/pos_embedding/RotaryPositionEmbedding",
        "rope_half_interleaved", "tilelang_rope", host,
    )
    # Select the library's existing PTO backend for its tile-level intrinsics.
    # Keep the original host function and kernel bodies, including pass configs.
    original_kernel = fn.__globals__["rope_kernel"]
    globals_copy = fn.__globals__.copy()
    globals_copy["rope_kernel"] = _runtime().jit(
        target="pto", pass_configs=globals_copy["pass_configs"],
    )(original_kernel.__wrapped__)
    fn = types.FunctionType(fn.__code__, globals_copy, fn.__name__, fn.__defaults__, fn.__closure__)
    seq, dim = x.shape
    base = case.params.get("base", 10000.)

    def run(x):
        # The manifest accepts x only, while the original library expects
        # frequency tables. Generate these on every call and count their cost.
        inv = 1. / (base ** (torch.arange(dim // 2, dtype=torch.float32,
                                        device=x.device) / (dim // 2)))
        angles = torch.arange(seq, dtype=torch.float32, device=x.device)[:, None] * inv[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        cos = torch.cat((cos, cos), dim=-1).view(seq, 1, dim)
        sin = torch.cat((sin, sin), dim=-1).view(seq, 1, dim)
        # The original API is in-place; clone each input and return the clone.
        return fn(x.clone().view(seq, 1, dim), sin, cos, layout="half").view_as(x)

    return Candidate(
        Implementation(run), "tilelang-ascend::rope_half_interleaved.tilelang_rope",
        "Existing PTO backend; NeoX -> half; singleton head axis; clone in-place input; frequency-table cost included.",
        source_hashes(Path(__file__), LIBRARIES / "tilelang-ascend" / host),
    )
