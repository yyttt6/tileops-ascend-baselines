"""Attempt original generic examples on Ascend without schedule edits."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes
from benchmarks.ascend_tilelang_baselines import _runtime, _runtime_hashes


def _original_jit(relative, name):
    tilelang = _runtime()
    source = LIBRARIES / "tilelang" / relative
    tree = ast.parse(source.read_text(), str(source))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(functions) == 1
    function = functions[0]
    # Bypass example autotuning and select backend at the host decorator.
    # Function body, original tile constants and pipeline are untouched.
    function.decorator_list = ast.parse('@tilelang.jit(target="ascend")\ndef f(): pass').body[0].decorator_list
    adapted = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    key = hashlib.sha256(adapted.encode()).hexdigest()[:20]
    cache = WORKSPACE / ".cache/ascend-classic" / ("tilelang-generic-" + key)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / source.name
    path.write_text(adapted)
    module_name = "_classic_generic_" + key
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, name), source_hashes(source, path) | _runtime_hashes(tilelang)


def tilelang_generic_conv(case):
    from benchmarks.api import Implementation

    fn, hashes = _original_jit("examples/convolution/example_convolution.py", "convolution")
    p = case.params
    pair = lambda v: (v, v) if isinstance(v, int) else tuple(v)
    s, d, padding = pair(p["stride"]), pair(p["dilation"]), pair(p["padding"])
    if s[0] != s[1] or d[0] != d[1] or padding[0] != padding[1] or p.get("groups", 1) != 1:
        raise UnsupportedCase("Original example uses equal spatial stride/dilation/padding and groups1.")

    def run(x, weight, *extra):
        if any(v is not None for v in extra):
            raise UnsupportedCase("Original example has no bias.")
        data = x.permute(0, 2, 3, 1).contiguous()
        kernel = weight.permute(2, 3, 1, 0).contiguous()
        out = fn(data, kernel, s[0], d[0], padding[0], 64, 128, 32, 3, 256)
        return out.permute(0, 3, 1, 2).contiguous()

    return Candidate(Implementation(run), "tilelang::examples/convolution.convolution",
                     "Original generic example, Ascend backend, NCHW/NHWC layout only; original main tiles/pipeline.",
                     source_hashes(Path(__file__)) | hashes)


def tilelang_generic_ssd(case):
    import torch
    from benchmarks.api import Implementation

    fn, hashes = _original_jit("examples/linear_attention/example_mamba_chunk_state.py", "chunk_state_fwd")

    def run(x, b, dt, da, seq_idx):
        if seq_idx is not None:
            raise UnsupportedCase("This original example uses dense chunks without sequence boundaries.")
        # Preserve the original example's public FP16 data path.
        return fn(b.to(torch.float16).contiguous(), x.to(torch.float16).contiguous(),
                  dt.to(torch.float16).contiguous(), da.to(torch.float16).contiguous()).float()

    return Candidate(
        Implementation(run), "tilelang::example_mamba_chunk_state.chunk_state_fwd",
        "Original chunk-state computation and default tiles/pipeline, targetAscend; public FP16 input/output casts.",
        source_hashes(Path(__file__)) | hashes,
        limitations=("FP32 dt/dA are rounded by the original FP16 API; numerical tolerance must still pass.",),
    )
