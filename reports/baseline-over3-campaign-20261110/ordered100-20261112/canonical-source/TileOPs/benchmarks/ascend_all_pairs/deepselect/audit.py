"""CPU-only source/metadata audit; never imports any device runtime or workload.

Usage: python3 -I -S path/to/deepselect/audit.py
Outputs coverage.json, progress.json, static_checks.json and source_catalog.json
in this package and task evidence directory. No numerical tests are fabricated.
"""

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
from math import prod
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
EVIDENCE = WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/deepselect"
INVENTORY = EVIDENCE.parent / "inventory.json"
sys.dont_write_bytecode = True


class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "torch_npu", "deep_select", "acl", "tilelang", "triton"}:
            raise AssertionError(f"CPU audit attempted forbidden runtime import: {fullname}")
        return None


sys.meta_path.insert(0, NoRuntime())
sys.path.insert(0, str(WORKSPACE / "TileOPs"))
from benchmarks.ascend_all_pairs.deepselect import PROVIDERS
from benchmarks.ascend_all_pairs.deepselect.loader import VENDOR, vendor_hashes
from benchmarks.ascend_all_pairs.deepselect.semantics import adaptive_windows, axes_plan, pool_plan, support


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).relative_to(WORKSPACE))


def save(name, value):
    # The memory directory is explicitly task-owned; shared memory is never changed.
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (HERE / name).write_text(text)
    (EVIDENCE / name).write_text(text)


def clean(value):
    if isinstance(value, float) and (value == float("inf") or value == float("-inf")):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def main():
    EVIDENCE.mkdir(exist_ok=True)
    inventory_bytes = INVENTORY.read_bytes()
    inventory = json.loads(inventory_bytes)
    inventory_hash = hashlib.sha256(inventory_bytes).hexdigest()
    (EVIDENCE / ("inventory_" + inventory_hash[:12] + ".json")).write_bytes(inventory_bytes)
    source_evidence = {}

    def source(key, path, start=1, end=None, meaning=None):
        path = Path(path)
        lines = path.read_text().splitlines()
        source_evidence[key] = {"file": rel(path), "line_start": start,
                                "line_end": min(end or len(lines), len(lines)), "sha256": sha(path), "meaning": meaning}
        return key

    source("api", VENDOR / "csrc/api.cpp", 31, 303, "Only TopK selection and alignment are exported; Ascend is BF16/int32, unsorted values.")
    source("interface", VENDOR / "deep_select/interface.py", 34, 124, "Original Python wrapper allocates aligned outputs and calls the native backend.")
    source("backend", VENDOR / "csrc/ascend_kernels/kernel.asc", 18, 590, "Original radix/histogram maximum selection, NaN policy and +0 > -0 ordering.")
    source("page_order", VENDOR / "csrc/ascend_kernels/config.h", 38, 89, "Long rows use shuffled scan pages; selected equal values need not have the lowest original index.")
    source("bounds", VENDOR / "csrc/structs.h", 24, 42, "Ascend 32-byte alignment and width < 2M bound.")
    source("adapters", HERE / "adapters.py", meaning="All input-dependent packing stays inside run; only original TopK computes maxima.")
    source("mapping", HERE / "semantics.py", meaning="Pure metadata constraints, reduction/pooling shape plans.")
    source("loader", HERE / "loader.py", meaning="Current architecture, original Python, source and binary hashes are checked at runtime.")
    source("setup", VENDOR / "setup.py", 171, 252, "Original Ascend build flags and source list; setup.py itself is NOT imported.")
    api_text = (VENDOR / "csrc/api.cpp").read_text()
    exports = re.findall(r'm\.def\("([^\"]+)"', api_text)
    assert exports == ["topk", "get_alignment_requirement"], exports
    assert set(PROVIDERS) == {"AmaxFwdOp", "AnyFwdOp", "MaximumFwdOp", "LogicalOrFwdOp", "ReluFwdOp",
                               "MaxPool1dFwdOp", "MaxPool2dFwdOp", "MaxPool3dFwdOp", "AdaptiveMaxPool2dFwdOp"}

    factory_sources = {}
    for path in sorted((WORKSPACE / "TileOPs/benchmarks/_cases").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ENTRIES" for t in node.targets):
                for constant in ast.walk(node.value):
                    if isinstance(constant, ast.Constant) and isinstance(constant.value, str) and constant.value.endswith("Op"):
                        factory_sources[constant.value] = source("factory:" + constant.value, path, constant.lineno, constant.end_lineno, "Literal/comprehension member in ENTRIES, confirmed by AST without importing workloads.")
    missing_factories = sorted(o["op"] for o in inventory["operators"] if o["op"] not in factory_sources)
    manifest_sources, ref_apis = {}, {}
    for op in inventory["operators"]:
        name = op["op"]
        path = WORKSPACE / "TileOPs" / op["manifest"]
        lines = path.read_text().splitlines()
        begin = next(i for i, line in enumerate(lines) if line == name + ":")
        end = next((i for i in range(begin + 1, len(lines)) if re.match(r"^[A-Za-z0-9_]+Op:\s*$", lines[i])), len(lines))
        manifest_sources[name] = source("manifest:" + name, path, begin + 1, end, "Unchanged manifest contract and workloads.")
        ref_apis[name] = next((line.split("ref_api:", 1)[1].split(" #", 1)[0].strip() for line in lines[begin:end] if re.match(r"  ref_api:", line)), None)

    known_reasons = {
        "AminFwdOp": "Native API selects largest values only; it has no smallest/ascending-value option. Negating input/output would add target arithmetic outside the permitted layout adaptation.",
        "MinimumFwdOp": "Native API selects largest values only; no minimum/ascending-value option exists. Returning one of a pair using a torch comparison would substitute target math.",
        "ArgminFwdOp": "Native API has no smallest selection and does not implement the required first-minimum index semantics.",
        "ArgmaxFwdOp": "Complete exact first-index semantics are not preserved: long rows shuffle scan pages; even short rows order +0 above -0 rather than selecting the first equal numeric zero. An index-only route without those guarantees is not registered.",
        "AllFwdOp": "All is a minimum over boolean truth values, not largest selection. Arbitrary input truth conversion also has no native API; a torch all/nonzero test would replace the target computation.",
        "CountNonzeroFwdOp": "TopK's internal histogram is not an exported nonzero-count output. No API computes arbitrary-value nonzero counts.",
        "TopKSelectFwdOp": "Manifest index_score is FP32; Ascend DeepSelect accepts only BF16. FP32-to-BF16 changes ranking and ties. CUDA FP32 support is not an Ascend implementation; starts/end layout alone cannot solve the dtype mismatch.",
        "TopKMaskFwdOp": "Requires full masked_logits and preserves all logits tied at the kth threshold with per-row tensor k. Native scalar-k TopK returns selected values/indices only; selecting k positions and scattering loses threshold ties, and torch threshold/comparison/masking would compute missing target math.",
        "TopPMaskFwdOp": "Requires probability normalization, cumulative mass cutoff and a full masked-logit result. Native TopK neither sorts values on Ascend nor computes cumulative probabilities.",
        "TopKTopPMaskFwdOp": "Requires combined per-row k selection, normalized cumulative top-p cutoff and a full tie-aware masked-logit output; native TopK lacks the latter mathematics.",
        "MinPMaskFwdOp": "Maximum is only an intermediate: min-p needs log/probability threshold arithmetic and full comparison/masking, none exported by DeepSelect.",
        "SamplingFromProbsFwdOp": "Probability sampling with seed/offset is stochastic categorical selection, not deterministic largest-value TopK; no RNG/CDF sampling API exists.",
        "ChainSpeculativeSamplingFwdOp": "Needs acceptance/rejection probability arithmetic, RNG and accepted-prefix outputs; no such original API exists.",
        "FusedTopKFwdOp": "MoE fused routing also requires scoring/normalization/bias/weights; raw largest-index selection does not return the complete manifest output.",
        "VectorNormFwdOp": "Even infinity norm requires absolute values before maximum selection. Native TopK returns original signed lanes and exposes neither abs nor other norm arithmetic.",
        "LogicalAndFwdOp": "Boolean conjunction is minimum in the 0/1 domain; native TopK exposes maximum only. Nonboolean truth conversion would also require non-native predicates.",
    }
    category_reasons = {
        "elementwise": "The reference requires arithmetic, a value transform, predicate, conditional selection, rounding or RNG beyond returning the largest unchanged lane. No owned Ascend entry exports this operation; embedded dependencies/test references are not providers.",
        "normalization": "Normalization requires moments/norms and affine or fused arithmetic; native TopK has no complete normalization output or backward API.",
        "gemm": "Requires multiply-accumulate matrix products (and any declared quantization); DeepSelect's owned Ascend API is selection only. Bundled CUTLASS is not a DeepSelect Ascend GEMM implementation.",
        "attention": "Requires attention/index-score arithmetic or dedicated cache/index movement with the complete declared outputs. A TopK substep or torch gather followed by an identity selection does not implement this operation in DeepSelect.",
        "linear_attention": "Requires recurrent/chunked state updates, matrix products or gradients; no owned DeepSelect API returns these states or outputs.",
        "moe": "Requires expert arithmetic, routing weights or permutation/alignment outputs absent from TopK; top-k indices alone are not the complete manifest output.",
        "quantization": "Requires scaling, rounding, clipping, packing/dequantization and possibly scales; original selected lanes/indices do not implement those outputs.",
        "reduction": "Requires arithmetic aggregation, scan or normalization beyond selecting the maximum. Internal TopK histogram/threshold state is not an exported equivalent reduction API.",
        "rope": "Requires position-dependent sinusoidal rotation/scaling; no original DeepSelect API implements rotary mathematics.",
        "sampling": "Requires sampling or full threshold-mask mathematics absent from the owned TopK API.",
        "ssm": "Requires state-space recurrence, convolution, cumulative transforms or state outputs; original TopK has none of these entry points.",
        "other": "Requires transform, sequence-model arithmetic, gating or gradients; DeepSelect exports only largest-value selection and alignment metadata.",
        "conv_pool": "Requires convolution multiply-accumulate or average/sum pooling. Original maximum selection cannot compute those arithmetic results.",
    }

    compile_dir = WORKSPACE / ".cache/all-pairs/deepselect/compile"
    compile_reports = []
    for stage in ("kernel", "api", "api_compat"):
        receipt = compile_dir / (stage + "_compile.json")
        if receipt.exists():
            shutil.copyfile(receipt, EVIDENCE / receipt.name)
            log = compile_dir / (stage + ".log")
            shutil.copyfile(log, EVIDENCE / log.name)
            report = json.loads(receipt.read_text())
            report["receipt_path"] = rel(EVIDENCE / receipt.name)
            report["log_path"] = rel(EVIDENCE / log.name)
            compile_reports.append(report)
    deployment = {
        "schema": "deepselect-deployment/v1", "status": "deployment_blocked", "target": "aarch64/CANN 9.1.0/dav-3510",
        "reason": "Original kernel.asc fails CPU compilation: CANN 9.1 headers do not declare asc_pack_to_low, asc_duplicate_highest, asc_squeeze_and_storeunalign_init, asc_squeeze_and_storeunalign, asc_squeeze_and_storeunalign_finalize. No NPU import or execution occurred; no extension was copied from another machine.",
        "source_unchanged": True, "device_initialized": False, "compile_reports": compile_reports,
        "host_api_compatibility": "Original api.cpp also failed on unavailable std::format at line 101. A private copy changes only the TORCH_CHECK diagnostic to its equivalent variadic formatting; api_compat.cpp compiles successfully. The selection kernel remains blocked, so no extension is linked or loaded.",
        "host_api_compatibility_source": rel(compile_dir / "api_compat.cpp"),
        "host_api_compatibility_sha256": sha(compile_dir / "api_compat.cpp") if (compile_dir / "api_compat.cpp").exists() else None,
        "compatibility_review": "CANN 9.1 has asc_pack/asc_pack_v2, but no inspected one-to-one public alias for all missing highest/fused squeeze-store operations. Reconstructing those sequences without the newer definitions risks changing kernel scheduling. No speculative shim was applied.",
    }
    save("deployment.json", deployment)

    operators, cases, geometry = [], [], []
    by_op = {o["op"]: o for o in inventory["operators"]}
    for entry in inventory["cases"]:
        name = entry["op"]
        params = entry.get("tensors", {}) | entry["params"]
        op = by_op[name]
        status, reason = support(name, params)
        if name not in PROVIDERS:
            if name.startswith("MaxPool") and "Indices" in name or name == "AdaptiveMaxPool2dIndicesFwdOp":
                reason = "Value windows can use TopK, but the complete output includes first-max spatial indices. Original +0 > -0 ordering does not preserve first-index ties for mixed signed zero; longer windows also use shuffled pages. No torch argmax/equality correction is substituted."
            else:
                reason = known_reasons.get(name, category_reasons[op["category"]])
            reason = f"{name} ({ref_apis[name] or 'manifest signature'}): " + reason
        sources = ["api", "interface", "backend", "bounds", manifest_sources[name]]
        if name in factory_sources:
            sources.append(factory_sources[name])
        if name in PROVIDERS:
            sources += ["adapters", "mapping", "loader"]
        if name == "ArgmaxFwdOp" or "MaxPool" in name and "Indices" in name:
            sources.append("page_order")
        row = {
            "op": name, "case_id": entry["case_id"], "params": clean(params),
            "status": "deployment_blocked" if status == "supported_unverified" else "unsupported",
            "semantic_status": status, "reason": reason,
            "deployment_status": "deployment_blocked" if status == "supported_unverified" else "not_applicable",
            "deployment_receipt_path": rel(HERE / "deployment.json") if status == "supported_unverified" else None,
            "adapter_path": rel(HERE / "adapters.py") if status == "supported_unverified" else None,
            "provider_registered": name in PROVIDERS,
            "factory_status": "present_static" if name in factory_sources else "missing_factory",
            "numerical_status": "not_run_cpu_only", "numerical_receipt_path": None,
            "input_protection_receipt_path": None, "runtime_ownership_receipt_path": None,
            "sources": sources,
        }
        cases.append(row)
        if status == "supported_unverified":
            shape = params.get("x", params.get("input"))[0]
            plan = {"op": name, "case_id": entry["case_id"], "output_shape": params["output"][0]}
            if name.startswith("MaxPool"):
                _, kernel, stride, pad, dilation, output, right = pool_plan(params)
                # Independently enumerate legal output-start positions from the pool contract.
                for n, k, s, p, d, out, r in zip(shape[2:], kernel, stride, pad, dilation, output, right):
                    extent = d * (k - 1) + 1
                    starts = list(range(0, n + 2 * p - extent + 1, s))
                    if params.get("ceil_mode", False) and starts and starts[-1] + extent < n + 2 * p:
                        next_start = starts[-1] + s
                        if next_start < n + p:
                            starts.append(next_start)
                    starts = [a for a in starts if a < n + p]
                    assert len(starts) == out, (entry["case_id"], len(starts), out)
                    assert (out - 1) * s + d * (k - 1) < n + p + r
                plan.update(packed_width=prod(kernel), padded_spatial=[n + p + r for n, p, r in zip(shape[2:], pad, right)])
            elif name.startswith("Adaptive"):
                _, (oh, ow), windows = adaptive_windows(params)
                h, w = shape[-2:]
                assert len(windows) == oh * ow
                assert all(0 <= hs < he <= h and 0 <= ws < we <= w for hs, he, ws, we in windows)
                plan["window_count"] = len(windows)
                plan["max_packed_width"] = max((he - hs) * (we - ws) for hs, he, ws, we in windows)
            elif name in ("AmaxFwdOp", "AnyFwdOp"):
                order, rows, width, output = axes_plan(shape, params.get("dim"), params.get("keepdim", False))
                assert sorted(order) == list(range(len(shape))) and rows * width == prod(shape)
                assert tuple(params["output"][0]) == output
                plan.update(packed_width=width, packed_rows=rows)
            else:
                plan["packed_width"] = 2
            geometry.append(plan)

    for op in inventory["operators"]:
        rows = [c for c in cases if c["op"] == op["op"]]
        supported = sum(c["semantic_status"] == "supported_unverified" for c in rows)
        operators.append({
            "op": op["op"], "category": op["category"], "family": op["family"],
            "case_count": len(rows), "semantic_supported_cases": supported,
            "unsupported_cases": len(rows) - supported,
            "status": "deployment_blocked" if supported else "unsupported",
            "semantic_status": "supported_unverified" if supported else "unsupported",
            "provider_registered": op["op"] in PROVIDERS,
            "adapter_path": rel(HERE / "adapters.py") if op["op"] in PROVIDERS else None,
            "source_evidence": list(dict.fromkeys(s for c in rows for s in c["sources"])),
            "case_status_counts": dict(Counter(c["status"] for c in rows)),
            "reason": "Per-case dtype/domain restrictions apply; see every case record." if supported else rows[0]["reason"],
            "numerical_receipt_path": None,
        })
    assert len(operators) == 205
    assert len(cases) == len(inventory["cases"])
    assert len({(c["op"], c["case_id"]) for c in cases}) == len(cases)
    assert sum(o["case_count"] for o in operators) == len(cases)

    parsed = []
    for file in sorted(HERE.glob("*.py")):
        ast.parse(file.read_text(), filename=str(file))
        compile(file.read_text(), str(file), "exec", dont_inherit=True)  # in-memory syntax compilation only
        parsed.append({"file": rel(file), "sha256": sha(file)})
    # Reject a torch implementation hidden behind the named baseline.
    adapter_tree = ast.parse((HERE / "adapters.py").read_text())
    torch_calls = []
    for node in ast.walk(adapter_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in ("torch", "F"):
                name = node.func.value.id + "." + node.func.attr
                assert name in {"torch.empty", "torch.empty_like", "F.pad"}, name
                torch_calls.append(name)
    checks = {
        "schema": "deepselect-static-checks/v1", "status": "passed", "kind": "source_and_metadata_only",
        "device_tests": 0, "numerical_tests": 0, "device_runtime_imports": [],
        "package_import_with_forbidden_runtime_finder": "passed",
        "syntax": parsed, "provider_keys": sorted(PROVIDERS), "vendor_exports": exports,
        "allowed_torch_calls_in_adapter": sorted(set(torch_calls)),
        "manifest_geometry": geometry,
        "coverage_unique_cases": len(cases), "coverage_operators": len(operators),
        "missing_factories_static": missing_factories,
        "inventory_missing_factories": inventory.get("missing_factory_ops", []),
        "manifest_hash_matches": {path: sha(WORKSPACE / path) == expected for path, expected in inventory["manifest_sha256"].items()},
        "limitations": ["No factory execution, tensor computation, native import, device access, numerical verification or benchmark.",
                        "Factory availability is AST presence in current ENTRIES, not successful runtime workload construction.",
                        "Compiler failure is retained separately and is not reported as a passed deployment check."],
    }
    assert all(checks["manifest_hash_matches"].values())
    assert not any(name in sys.modules for name in ("torch", "torch_npu", "deep_select", "acl", "tilelang", "triton"))
    save("static_checks.json", checks)
    catalog = {"schema": "deepselect-source-catalog/v1", "vendor_sha256": vendor_hashes(), "source_evidence": source_evidence,
               "owned_export_surface": exports, "excluded": ["CUDA backend FP32 support is not Ascend support.", "Tests/reference implementations and bundled CUTLASS/kerutils are not independently attributed operators."],
               "source_files_reviewed": sorted(str(p.relative_to(VENDOR)) for p in VENDOR.rglob("*") if p.is_file() and "3rdparty" not in p.parts and p.suffix in (".py", ".h", ".cuh", ".cpp", ".asc", ".cu"))}
    save("source_catalog.json", catalog)
    coverage = {
        "schema": "deepselect-all-pairs-coverage/v1", "library": "DeepSelect",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inventory": {"path": rel(INVENTORY), "sha256": inventory_hash, "operator_count": len(operators), "case_count": len(cases),
                      "assignment_case_count": 1982, "scope_note": "Shared inventory was expanded during this task from 1982 to 2062 rows to include missing-factory manifest cases. This audit covers every row of the current source-fingerprinted inventory."},
        "status_schema": {
            "status": "Current acceptance: deployment_blocked for implemented but unbuildable native routes; unsupported for no complete legal mapping in this source snapshot. Never means permanent impossibility.",
            "semantic_status": "supported_unverified means concrete complete-output adapter exists but no numerical/device receipt; unsupported means an explicit mathematical, dtype or exact-output mismatch.",
            "sources": "Each case sources[] and operator source_evidence[] resolves in source_evidence to file, 1-based line range and SHA256.",
            "factory_status": "Independent AST check: present_static or missing_factory. Missing workloads are not invented.",
            "numerical_status": "not_run_cpu_only for every row; numerical_failure and verified remain zero because no device call was attempted.",
            "params": "Manifest metadata, not generated tensors; nonfinite scalar parameters are encoded as the strings Infinity/-Infinity for strict JSON.",
        },
        "summary": {"operators": len(operators), "cases": len(cases), "providers": len(PROVIDERS),
                    "semantic_supported_unverified_cases": sum(c["semantic_status"] == "supported_unverified" for c in cases),
                    "case_status_counts": dict(Counter(c["status"] for c in cases)), "verified": 0, "numerical_failure": 0,
                    "missing_factory_ops": missing_factories, "missing_factory_cases": sum(c["factory_status"] == "missing_factory" for c in cases)},
        "source_evidence": source_evidence, "operators": operators, "cases": cases,
        "static_receipt_path": rel(HERE / "static_checks.json"), "deployment_receipt_path": rel(HERE / "deployment.json"),
        "historical_evidence": {"path": "TileOPs/benchmarks/ascend/classic_matrix.json", "used_as_current_device_result": False,
                                "note": "Historical x86/CANN9.2 verified records are read-only leads, not ARM64 results."},
        "oracle_issues": [],
    }
    save("coverage.json", coverage)
    progress = {
        "schema": "deepselect-progress/v1", "worker": "0aca6ad43b1a4553", "owner": "allpairs-remote/deepselect",
        "updated_at": coverage["generated_at"], "phase": "cpu_scope_complete", "device_execution": "forbidden", "device_tests": 0,
        "summary": coverage["summary"], "providers": sorted(PROVIDERS),
        "changed_files": sorted(rel(p) for p in HERE.iterdir() if p.is_file()),
        "remaining_issues": [deployment["reason"],
                             "Future authorized device run must verify all manifest outputs, preserved inputs, and actual owned native calls for each supported case; no such claim is made now.",
                             "Shared factory availability changed during task: current source AST and inventory flags are both retained for root reconciliation.",
                             "NaN propagation and signed-zero first-index selection are not repaired by fallback computation; index-output routes remain unregistered."],
        "checks": rel(HERE / "static_checks.json"), "coverage": rel(HERE / "coverage.json"),
        "notifications": "Supervisor persists final response and notifies parent; worker sends no external notification.",
    }
    save("progress.json", progress)
    with (EVIDENCE / "audit_history.jsonl").open("a") as history:
        history.write(json.dumps({"at": coverage["generated_at"], "inventory_sha256": inventory_hash,
                                  "summary": coverage["summary"], "coverage_sha256": sha(HERE / "coverage.json")}) + "\n")
    print(json.dumps(coverage["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
