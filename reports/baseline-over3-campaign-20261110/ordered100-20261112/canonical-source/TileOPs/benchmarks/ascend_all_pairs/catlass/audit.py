"""Generate source-evidenced coverage using only the Python standard library.

Does not construct Case inputs, import torch/vendor code, load shared libraries,
execute a factory or access a device. Numeric receipts are necessarily null.
"""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
VENDOR = WORKSPACE / "ascend_baselines/catlass"
EVIDENCE = WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/catlass"
INVENTORY = EVIDENCE.parent / "inventory.json"
CACHE = WORKSPACE / ".cache/all-pairs/catlass"


def load_package():
    name = "_catlass_cpu_audit"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "__init__.py", submodule_search_locations=[str(HERE)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(WORKSPACE))


def ref(path, marker=None, *, line=None):
    lines = path.read_text().splitlines()
    if line is None:
        line = next((i + 1 for i, value in enumerate(lines) if marker in value), None) if marker else 1
    if line is None:
        raise ValueError(f"Source marker absent: {path}: {marker}")
    return {"file": rel(path), "line": line, "sha256": digest(path), "excerpt": lines[line - 1].strip()}


# Markers intentionally point to original definitions/options, not historical logs.
SOURCES = {
    "gemm": ("tests/optest/kernels/43_ascend950_basic_matmul/ascend950_basic_matmul_impl.cpp", "using ElementA"),
    "gemm_kernel": ("include/catlass/gemm/kernel/basic_matmul_tla.hpp", "class BasicMatmul"),
    "gemm_bias": ("examples/43_ascend950_basic_matmul/basic_matmul_tla.cpp", "using ElementBias"),
    "fp8": ("tests/optest/kernels/51_ascend950_quant_matmul_per_group_per_block_tla/quant_matmul_per_group_per_block_impl.cpp", "using ElementA"),
    "fp8_epilogue": ("include/catlass/epilogue/block/block_epilogue_per_group_per_block.hpp", "IsSameType<CType, int32_t>"),
    "int8": ("tests/optest/kernels/57_ascend950_matmul_full_dequant/matmul_full_dequant_impl.cpp", "using ElementA"),
    "int8_epilogue": ("include/catlass/epilogue/block/block_epilogue_dequant.hpp", "class BlockEpilogue"),
    "attention": ("examples/49_ascend950_flash_attention_infer/fai_kernel.h", "using L1TileShape = tla::Shape<_128, _128, _128>"),
    "attention_tiling": ("examples/49_ascend950_flash_attention_infer/fai_tiling.h", "uint32_t maskType"),
    "attention70": ("examples/70_ascend950_flash_attention_chunk_prefill/fai_kernel.cpp", "using L1TileShape"),
    "dsl_unary": ("python/tla_dsl/examples/end_to_end/vector_ops/unary_ops.py", "_UNMASKED_UNARY_OPS"),
    "dsl_binary": ("python/tla_dsl/examples/end_to_end/vector_ops/binary_op.py", "def _operator_specs"),
    "dsl_compare": ("python/tla_dsl/examples/end_to_end/vector_ops/compare_mask.py", '"vector_scalar_gt"'),
    "dsl_bitwise": ("python/tla_dsl/examples/end_to_end/vector_ops/bitwise_ops.py", "def bitwise_ops"),
    "dsl_sum": ("python/tla_dsl/examples/end_to_end/simt/reduction_simt/reduction_simt.py", "def reduction_simt"),
    "dsl_reduce": ("python/tla_dsl/examples/end_to_end/vector_ops/reduction_ops.py", "TILE_ELE = 64"),
    "dsl_multi": ("python/tla_dsl/examples/end_to_end/simt/multi_ops_simt/multi_ops_simt.py", "def "),
    "api_catalog": ("tests/optest/torch_catlass/ops/__init__.py", "from "),
    "moe_mx": ("examples/65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant/fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant.cpp", "using Element"),
    "moe_finalize": ("examples/71_ascend950_fp8_mx_grouped_matmul_finalize_routing/fp8_mx_grouped_matmul_finalize_routing.cpp", "using Element"),
    "grouped_native": ("examples/60_ascend950_grouped_matmul_slice_m/grouped_matmul_slice_m.cpp", "using Element"),
    "w8a16": ("examples/30_w8a16_matmul/w8a16_matmul.cpp", "using Element"),
    "w4a8": ("examples/32_w4a8_matmul/w4a8.cpp", "using Element"),
    "mx_w4": ("examples/74_ascend950_weight_quant_a8w4_grouped_mx_matmul/weight_quant_a8w4_grouped_mx_matmul.cpp", "using Element"),
    "mx_attention": ("experimental/attention/ascend950_fp8_mx_flash_attention_infer/fai_kernel.h", "using ElementQ"),
    "fag": ("tests/optest/kernels/87_fag_tla/fag_tla.cpp", "fagInfo.maskType = 0"),
    "fag_lse": ("include/catlass/epilogue/block/block_epilogue_fag_op.hpp", "AscendC::Exp("),
    "fag_cmake": ("tests/optest/kernels/87_fag_tla/CMakeLists.txt", "NPU_ARCH_LIST"),
    "backward_design": ("examples/105_ascend950_fmha_backward/ascend950_fmha_bwd.md", None),
    "mla": ("examples/19_mla/mla_kernel.cpp", "GM_ADDR"),
    "mla_host": ("tests/optest/kernels/19_mla/mla.cpp", "void Mla("),
    "mla_cmake": ("tests/optest/kernels/19_mla/CMakeLists.txt", "NPU_ARCH_LIST 2201"),
    "mla_dispatch": ("include/catlass/gemm/block/block_mmad.hpp", "CATLASS_ARCH == 2201"),
    "mla_design": ("examples/107_ascend950_mla/mla_design.md", None),
    "ssd_design": ("examples/108_ascend950_mamba_ssd/ssd_design.md", None),
    "x_attention": ("examples/78_x_attention/x_attention_kernel.hpp", "class "),
    "rain_attention": ("experimental/attention/ascend950_rain_fusion_attention/rfa_kernel.cpp", "using Element"),
    "activation_silu": ("include/catlass/epilogue/tile/tile_elemwise_silu.hpp", "struct "),
    "activation_gelu": ("include/catlass/epilogue/tile/tile_elemwise_gelu.hpp", "struct "),
    "conv_native": ("examples/56_ascend950_basic_conv2d_tla/basic_conv2d_tla.cpp", "using Element"),
    "quant_output": ("examples/62_ascend950_broadcast_matmul_perblock_quant/broadcast_matmul_perblock_quant_tla.cpp", "using Element"),
    "complex_matmul": ("examples/77_planar_complex_matmul/planar_complex_matmul.cpp", "using Element"),
}


def write_json(name, value, mirror=True):
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    (HERE / name).write_text(text)
    if mirror:
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        (EVIDENCE / name).write_text(text)


def source_catalog():
    roots = ["examples", "experimental", "include", "tests/optest", "python/tla_dsl/catlass", "python/tla_dsl/examples"]
    files = []
    for root in roots:
        for path in sorted((VENDOR / root).rglob("*")):
            if path.is_file() and path.suffix in (".cpp", ".h", ".hpp", ".py", ".md", ".txt", ".td"):
                entry = {"file": rel(path), "sha256": digest(path), "bytes": path.stat().st_size}
                if path.suffix == ".py":
                    tree = ast.parse(path.read_text())
                    entry["definitions"] = [{"name": n.name, "line": n.lineno} for n in tree.body
                                            if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))]
                files.append(entry)
    return {"schema": "catlass-source-catalog/v1", "roots": roots,
            "excludes": "third-party dependencies, compiled binaries, history, hidden VCS data",
            "files": files, "file_count": len(files)}


def adapter_ref(package, op):
    if op in package._NATIVE_ROUTES:
        module, function = package._NATIVE_ROUTES[op]
    elif op == "SumFwdOp":
        module, function = "reduction", "sum_reduction"
    elif op == "MLADecodeWithKVCacheFwdOp":
        module, function = "mla", "mla_decode"
    elif op in ("MaxPool1dFwdOp", "MaxPool2dFwdOp", "MaxPool3dFwdOp"):
        module, function = "pooling", "max_pool"
    elif op == "AdaptiveMaxPool2dFwdOp":
        module, function = "pooling", "adaptive_max_pool"
    elif op in package._VECTOR_OPS:
        module, function = "vector", "vector"
    else:
        return None
    return ref(HERE / f"{module}.py", f"def {function}(") | {"factory": f"{module}.{function}"}


def build_receipts():
    from importlib import import_module
    builder = import_module(load_package().__name__ + ".build")
    result = {}
    for kind in builder.KINDS:
        hashes_now = builder.hashes(builder.sources(kind))
        candidates = []
        for path in (CACHE / "native").glob(f"{kind}-*/build.json"):
            value = json.loads(path.read_text())
            if value.get("kind") == kind and value.get("source_sha256") == hashes_now:
                candidates.append((path, value))
        candidates.sort(key=lambda x: x[0].stat().st_mtime_ns)
        if not candidates:
            result[kind] = {"status": "unbuilt", "reason": "No compile receipt matches all current source hashes"}
            continue
        path, receipt = candidates[-1]
        binary = path.parent / "libcatlass_all_pairs.so"
        ok = receipt["returncode"] == 0 and binary.is_file() and digest(binary) == receipt.get("binary_sha256")
        result[kind] = {"status": "compiled_not_loaded" if ok else "compile_failed",
                        "receipt": rel(path), "receipt_sha256": digest(path),
                        "log": rel(Path(receipt["log"])), "returncode": receipt["returncode"],
                        "machine": receipt["machine"], "sdk": receipt["sdk"],
                        "binary": rel(binary) if ok else None,
                        "binary_sha256": receipt.get("binary_sha256"),
                        "device_loaded": False, "device_run": False}
    return result


def main():
    from importlib import import_module
    package = load_package()
    policy = import_module(package.__name__ + ".policy")
    inventory_bytes = INVENTORY.read_bytes()
    inventory = json.loads(inventory_bytes)
    inventory_hash = hashlib.sha256(inventory_bytes).hexdigest()
    # The full input snapshot is preserved; do not rewrite source manifests.
    (HERE / "inventory_snapshot.json").write_bytes(inventory_bytes)
    sources = {key: ref(VENDOR / path, marker) for key, (path, marker) in SOURCES.items()}
    sources["nsa_contract"] = ref(WORKSPACE / "TileOPs/workloads/attention/nsa.py", "class NSACompressed")
    sources["bwd_contract"] = ref(WORKSPACE / "TileOPs/workloads/attention/gqa/bwd.py", "math.log2(math.e)")
    catalog = source_catalog()
    write_json("source_catalog.json", catalog)
    ops_by_name = {op["op"]: op for op in inventory["operators"]}
    missing = set(inventory["missing_factory_ops"])
    native_receipts = build_receipts()
    legacy_builder = import_module(package.__name__ + ".legacy_build")
    current_legacy_sources = legacy_builder.hashes(legacy_builder.sources())
    legacy_receipt = None
    for path in (CACHE / "legacy").glob("mla-*/build.json"):
        report = json.loads(path.read_text())
        if report["source_sha256"] == current_legacy_sources:
            errors = [line for line in Path(report["log"]).read_text().splitlines() if "error:" in line]
            legacy_receipt = {"status": "compile_failed" if report["returncode"] else "compiled_not_loaded",
                              "receipt": rel(path), "receipt_sha256": digest(path), "log": rel(Path(report["log"])),
                              "returncode": report["returncode"], "first_error": errors[0] if errors else None,
                              "device_loaded": False, "device_run": False}
    if legacy_receipt is None:
        raise RuntimeError("Current MLA source requires a CPU compatibility-build receipt")
    cases, operators = [], []
    for case in inventory["cases"]:
        op = case["op"]
        decision = policy.classify(op, case["params"], case["family"])
        compatible = decision["semantic_status"] == "supported"
        registered = op in package.PROVIDERS
        status = "supported_unverified" if compatible and registered else "deployment_blocked" if compatible else "unsupported"
        detail = decision["reason"]
        blocker = None
        if compatible and not registered:
            if decision["route"] == "mla":
                blocker = {"code": "original_atlasa2_dispatch_unavailable_for3510", **legacy_receipt,
                           "reason": "Original MLA complete semantics match; unchanged AtlasA2 dispatch cannot instantiate with CATLASS_ARCH=3510. No architecture-tag/schedule substitution applied."}
            else:
                blocker = {"code": "missing_original_dsl_bindings", "path": rel(VENDOR / "python/tla_dsl/catlass/_mlir"),
                           "exists": (VENDOR / "python/tla_dsl/catlass/_mlir").exists(),
                           "reason": "Original generated MLIR Python/native bindings absent. No DSL or torch import/compile/run attempted."}
        manifest = WORKSPACE / "TileOPs" / ops_by_name[op]["manifest"]
        cases.append({"op": op, "case_id": case["case_id"], "case_key": op + "::" + case["case_id"],
                      "family": case["family"], "status": status, **decision,
                      "reason": detail, "sources": [sources[key] for key in decision["source_keys"]],
                      "manifest_source": ref(manifest, op), "params": case["params"],
                      "adapter": adapter_ref(package, op) if compatible else None,
                      "provider_registered": registered, "case_accepted_by_provider": compatible and registered,
                      "workload_factory_available": op not in missing,
                      "deployment_blocker": blocker,
                      "numerical_status": "not_run_device_forbidden", "numerical_receipt": None,
                      "full_output_check": "pending_original_manifest_oracle", "input_protection_check": "pending_device_harness",
                      "runtime_call_ownership_check": "pending_device_harness", "device_runs": 0})
    for op in inventory["operators"]:
        own = [case for case in cases if case["op"] == op["op"]]
        counts = dict(Counter(case["status"] for case in own))
        accepted = [case for case in own if case["semantic_status"] == "supported"]
        status = ("supported_unverified" if counts.get("supported_unverified") else "deployment_blocked"
                  if counts.get("deployment_blocked") else "unsupported")
        all_keys = sorted({key for case in own for key in case["source_keys"]})
        operators.append({**op, "status": status, "case_status_counts": counts,
                          "partial_support": bool(accepted) and len(accepted) != len(own),
                          "provider_registered": op["op"] in package.PROVIDERS,
                          "workload_factory_available": op["op"] not in missing,
                          "adapter": adapter_ref(package, op["op"]) if accepted else None,
                          "reasons": sorted({case["reason"] for case in own}),
                          "sources": [sources[key] for key in all_keys],
                          "case_keys": [case["case_key"] for case in own], "numerical_receipt": None})
    now = datetime.now(timezone.utc).isoformat()
    coverage = {
        "schema": "catlass-all-pairs-coverage/v1", "schema_description": {
            "operators": "One row per exact manifest op; aggregate status prefers any supported case; partial_support and case_status_counts retain mixed outcomes.",
            "cases": "One row per original inventory case, exact parameters preserved. sources/manifest_source carry workspace-relative file,1-based line,SHA256. adapter is null only for unsupported cases.",
            "supported_unverified": "Matching complete adapter registered and original native templates CPU-compiled; no runtime/numeric/input protection validation.",
            "deployment_blocked": "Matching complete adapter implemented but original DSL generated bindings absent or native AtlasA2-to3510 compatibility compile failed; excluded from PROVIDERS.",
            "unsupported": "No equivalent complete original entry under allowed adaptations for this case in inspected snapshot; specific mathematical/type/parameter/output/architecture reason. Not a permanent impossibility claim.",
            "numerical_failed": "Reserved for an actual failed original-manifest run; none executed here.",
            "verified": "Reserved for actual full-output oracle + input protection + runtime ownership receipts; none executed here.",
            "factory_missing": "Orthogonal workload_factory_available=false, never presented as a device test or inferred numeric failure."},
        "worker": "a1853b6299bc4c8a", "owner": "allpairs-remote/catlass", "library": "catlass", "updated_at": now,
        "scope": {"initial_requested_ops": 205, "initial_factory_cases": 1982, "operators": len(operators),
                  "manifest_cases": len(cases), "factory_case_count": sum(c["workload_factory_available"] for c in cases),
                  "inventory": rel(INVENTORY), "inventory_sha256": inventory_hash,
                  "inventory_snapshot": rel(HERE / "inventory_snapshot.json"), "manifest_sha256": inventory["manifest_sha256"]},
        "environment": {"machine": platform.machine(), "device_access": "forbidden", "current_host_numeric_runs": 0,
                        "vendor_imports": 0, "native_libraries_loaded": 0, "historical_verified_reused": 0},
        "summary": {"operator_status_counts": dict(Counter(o["status"] for o in operators)),
                    "case_status_counts": dict(Counter(c["status"] for c in cases)),
                    "providers": len(package.PROVIDERS), "deferred_providers": len(package.DEFERRED_PROVIDERS),
                    "verified": 0, "numerical_failed": 0, "device_runs": 0},
        "missing_workload_factory_ops": sorted(missing), "original_missing_factory_ops": inventory.get("missing_factory_ops_initial", []),
        "source_catalog": {"file": rel(HERE / "source_catalog.json"), "sha256": digest(HERE / "source_catalog.json"),
                           "files": catalog["file_count"]},
        "source_evidence": sources, "native_builds": native_receipts, "legacy_compatibility_build": legacy_receipt,
        "oracle_suspicions": [], "static_receipt": rel(HERE / "static_checks.json"),
        "operators": operators, "cases": cases,
    }
    write_json("coverage.json", coverage)
    progress = {"schema": "catlass-all-pairs-progress/v1", "worker": "a1853b6299bc4c8a", "owner": "allpairs-remote/catlass",
                "phase": "coverage_generated_static_checks_pending", "updated_at": now,
                "scope": coverage["scope"], "summary": coverage["summary"], "npu_allowed": False,
                "device_runs": 0, "verified": 0, "changed_files": sorted(rel(p) for p in HERE.rglob("*") if p.is_file()),
                "native_compile_results": native_receipts, "legacy_compatibility_build": legacy_receipt, "missing_factory_ops": sorted(missing),
                "remaining": ["CPU static and provenance verification", "Future separately authorized full numeric/input/ownership checks",
                              "Original DSL generated bindings missing; deferred providers excluded", f"{len(missing)} shared workload factories missing"],
                "observations": ["Task-local source/evidence/cache writes only; no vendor/shared registry/manifest/oracle changes.",
                                 "Complete205-op inventory screened; latest inventory includes2062 manifest cases vs initial1982 factory cases.",
                                 "No original historical verified rows reused as current ARM64 results."]}
    write_json("progress.json", progress)
    print(json.dumps(coverage["summary"], indent=2))


if __name__ == "__main__":
    main()
