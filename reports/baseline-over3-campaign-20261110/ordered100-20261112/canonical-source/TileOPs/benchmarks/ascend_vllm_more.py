"""Small bindings to vllm-ascend's original Python kernels."""
import ast
from pathlib import Path
import sys
import types

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, source_hashes,
)
from benchmarks.ascend_transform_baselines import _rope_tables


def _original_module(relative, injected):
    """Resolve framework-only imports without loading the vLLM serving stack.

    Function bodies and decorators remain the original AST. Triton is the
    installed Ascend Triton; hardware helpers and env parsing are original
    vllm-ascend source, not replacement kernels or invented hardware values.
    """
    path = LIBRARIES / "vllm-ascend" / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    bypass = {"vllm.triton_utils", "vllm_ascend",
              "vllm_ascend.ops.triton.triton_utils"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in bypass:
            names = {entry.asname or entry.name for entry in node.names}
            if not names.issubset(injected):
                raise RuntimeError(f"Missing original import binding: {names - injected.keys()}")
        else:
            body.append(node)
    tree.body = body
    name = "_classic_vllm_" + relative.replace("/", "_").replace(".", "_")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(injected)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def _triton_bindings():
    import triton
    import triton.language as tl

    return {"triton": triton, "tl": tl, "HAS_TRITON": True}


def vllm_rope(case):
    import torch
    from benchmarks.api import Implementation

    x, = case.inputs
    if x.ndim != 2 or case.params["rope_layout"] != "neox":
        raise UnsupportedCase("This pilot uses a 1D NeoX input.")
    host = "vllm_ascend/ops/triton/rope.py"
    bindings = _triton_bindings()
    envs = _original_module("vllm_ascend/envs.py", {})
    utils = _original_module("vllm_ascend/ops/triton/triton_utils.py",
                             bindings | {"envs": envs})
    utils.init_device_properties_triton()
    module = _original_module(host, bindings | {
        "get_ub_size_bytes": utils.get_ub_size_bytes,
        "get_vectorcore_num": utils.get_vectorcore_num,
    })
    fn = module.rope_forward_triton_siso
    base = case.params.get("base", 10000.)

    def run(x):
        cos, sin = _rope_tables(torch, x, base)
        # The original API mutates its input, so preserve the manifest input.
        private = x.clone().unsqueeze(1)
        return fn(private, cos=cos, sin=sin, rope_dim=x.shape[-1],
                  is_neox_style=True).squeeze(1)

    return Candidate(
        Implementation(run), "vllm-ascend::rope_forward_triton_siso",
        "Singleton head, private copy for the original in-place API, per-call frequency tables.",
        source_hashes(Path(__file__), Path(__file__).with_name("ascend_transform_baselines.py"),
                      LIBRARIES / "vllm-ascend" / host,
                      LIBRARIES / "vllm-ascend/vllm_ascend/ops/triton/triton_utils.py",
                      LIBRARIES / "vllm-ascend/vllm_ascend/envs.py"),
    )


def vllm_gemm(case):
    from benchmarks.api import Implementation

    host = "vllm_ascend/ops/triton/batch_invariant/matmul.py"
    fn = _original_module(host, _triton_bindings()).matmul_persistent
    ta, tb = case.params["trans_a"], case.params["trans_b"]

    def run(a, b):
        return fn((a.T if ta else a).contiguous(),
                  (b.T if tb else b).contiguous(), bias=None)

    return Candidate(
        Implementation(run), "vllm-ascend::matmul_persistent",
        "Original BF16 GEMM without bias; manifest transpose flags and contiguous inputs only.",
        source_hashes(Path(__file__), LIBRARIES / "vllm-ascend" / host),
    )
