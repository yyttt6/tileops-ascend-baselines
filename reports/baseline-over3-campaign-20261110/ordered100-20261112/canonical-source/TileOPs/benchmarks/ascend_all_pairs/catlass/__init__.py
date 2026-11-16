"""CATLASS all-pairs providers. Import is stdlib-only and never initializes NPU.

Only concrete implemented routes are registered. Source-complete vector routes
remain in DEFERRED_PROVIDERS while the copied vendor DSL lacks native bindings.
Factories must only be invoked by a separately device-authorized harness.
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from benchmarks.api import Case
    from benchmarks.ascend_baselines import Candidate


def _factory(module, name, op, *, vector=False):
    def factory(case):
        from importlib import import_module
        from benchmarks.ascend_baselines import UnsupportedCase
        from .policy import classify
        decision = classify(op, case.params)
        if decision["semantic_status"] != "supported":
            raise UnsupportedCase(decision["reason"])
        target = getattr(import_module(f"{__name__}.{module}"), name)
        return target(case, op=op) if vector else target(case)
    factory.__name__ = f"catlass_{op}"
    factory.__doc__ = f"Original CATLASS adapter for {op}; see coverage.json for each case's status."
    return factory


_NATIVE_ROUTES = {
    "GemmFwdOp": ("gemm", "gemm"),
    "BmmFwdOp": ("gemm", "bmm"),
    "GroupedGemmFwdOp": ("gemm", "grouped_gemm"),
    "MoEGroupedGemmFwdOp": ("gemm", "moe_grouped_gemm"),
    "Conv1dFwdOp": ("convolution", "convolution"),
    "Conv2dFwdOp": ("convolution", "convolution"),
    "Conv3dFwdOp": ("convolution", "convolution"),
    "GQADenseFwdOp": ("attention", "dense"),
    "GQAVarlenFwdOp": ("attention", "varlen"),
    "GQAPagedFwdOp": ("attention", "paged"),
    "MHADecodePagedWithKVCacheFwdOp": ("attention", "mha_paged"),
    "GQAPrefillPagedWithKVCacheFwdOp": ("attention", "prefill_append"),
    "NSAVarlenFwdOp": ("attention", "nsa_selected"),
    "GemmFP8FwdOp": ("fp8", "gemm_fp8"),
    "BmmFP8FwdOp": ("fp8", "bmm_fp8"),
    "MoEGroupedGemmFP8FwdOp": ("fp8", "moe_grouped_fp8"),
    "GemmINT8W8A8FwdOp": ("int8", "int8_gemm"),
}
_VECTOR_OPS = (
    "AbsFwdOp", "NegFwdOp", "ExpFwdOp", "LogFwdOp", "SqrtFwdOp",
    "AddFwdOp", "SubFwdOp", "MulFwdOp", "DivFwdOp", "MaximumFwdOp", "MinimumFwdOp", "ReciprocalFwdOp",
    "BitwiseNotFwdOp", "BitwiseAndFwdOp", "BitwiseOrFwdOp", "BitwiseXorFwdOp",
    "LogicalNotFwdOp", "LogicalAndFwdOp", "LogicalOrFwdOp", "ReluFwdOp",
)

PROVIDERS: dict[str, Callable[["Case"], "Candidate"]] = {
    op: _factory(module, name, op) for op, (module, name) in _NATIVE_ROUTES.items()
}
DEFERRED_PROVIDERS: dict[str, Callable[["Case"], "Candidate"]] = {
    op: _factory("vector", "vector", op, vector=True) for op in _VECTOR_OPS
}
DEFERRED_PROVIDERS["SumFwdOp"] = _factory("reduction", "sum_reduction", "SumFwdOp")
for _op in ("MaxPool1dFwdOp", "MaxPool2dFwdOp", "MaxPool3dFwdOp"):
    DEFERRED_PROVIDERS[_op] = _factory("pooling", "max_pool", _op)
DEFERRED_PROVIDERS["AdaptiveMaxPool2dFwdOp"] = _factory("pooling", "adaptive_max_pool", "AdaptiveMaxPool2dFwdOp")
_bindings = Path(__file__).resolve().parents[4] / "ascend_baselines/catlass/python/tla_dsl/catlass/_mlir"
if _bindings.is_dir() and any(_bindings.rglob("*.so")):
    PROVIDERS.update(DEFERRED_PROVIDERS)
    DEFERRED_PROVIDERS = {}
# The original AtlasA2 dispatch is absent when CATLASS_ARCH=3510. Keep the
# concrete MLA adapter available for review, but never register a known failed build.
DEFERRED_PROVIDERS["MLADecodeWithKVCacheFwdOp"] = _factory("mla", "mla_decode", "MLADecodeWithKVCacheFwdOp")

__all__ = ["PROVIDERS", "DEFERRED_PROVIDERS"]
