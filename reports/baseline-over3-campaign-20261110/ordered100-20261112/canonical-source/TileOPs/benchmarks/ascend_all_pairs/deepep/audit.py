"""Reproducible CPU-only source audit of all 205 ops and the current inventory.

Run with ``python3 -B -S .../deepep/audit.py``. Only the standard library and
this package's inert provider registry are imported. No workloads are executed.
"""

import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.abc
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
sys.path.insert(0, str(WORKSPACE / "TileOPs"))
FORBIDDEN = {"torch", "torch_npu", "deep_ep", "deep_jit", "acl", "numpy", "tileops"}


class NoDeviceImports(importlib.abc.MetaPathFinder):
    attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            self.attempts.append(fullname)
            raise AssertionError(f"CPU audit attempted a forbidden import: {fullname}")


guard = NoDeviceImports()
sys.meta_path.insert(0, guard)
from benchmarks.ascend_all_pairs.deepep import PROVIDERS
from benchmarks.ascend_all_pairs.deepep.contracts import block_owners, paged_indices, plan
from benchmarks.ascend_all_pairs.deepep.runtime import digest, source_snapshot, vendor_sources

INVENTORY = WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/inventory.json"
EVIDENCE = WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/deepep"
VENDOR = WORKSPACE / "ascend_baselines/DeepEP-Ascend"
ADAPTER = "TileOPs/benchmarks/ascend_all_pairs/deepep/adapters.py"


def relative(path):
    return str(path.relative_to(WORKSPACE))


def source(path, anchor=None, end=None):
    path = Path(path)
    if not path.is_absolute():
        path = VENDOR / path
    lines = path.read_text().splitlines()
    number = 1 if anchor is None else next(i + 1 for i, line in enumerate(lines) if anchor in line)
    return {"file": relative(path), "line": number,
            "line_end": end or number, "sha256": digest(path)}


def manifest_source(operator):
    path = WORKSPACE / "TileOPs" / operator["manifest"]
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line == operator["op"] + ":")
    stop = next((i for i in range(start + 1, len(lines)) if re.match(r"^\w+Op:", lines[i])), len(lines))
    return source(path, operator["op"] + ":", stop)


API = source("csrc/python_api.cpp", "PYBIND11_MODULE", 74)
DISPATCH = [source("deep_ep/buffers/ep.py", "def dispatch("),
            source("csrc/buffers/ep.hpp", "dispatch(const torch::Tensor& x"),
            source("deep_ep/include/deep_ep/impls/ep/dispatch.hpp", "// TODO: the results to host may be not aligned", 532),
            source("deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.hpp", "void dispatch_copy_epilogue_impl(", 614)]
COMBINE = [source("deep_ep/buffers/ep.py", "def combine("),
           source("deep_ep/include/deep_ep/impls/ep/combine.hpp", "// copy topk weights", 112),
           source("deep_ep/include/deep_ep/impls/ep/combine.hpp", "if (num_slots >= 2)", 329),
           source("deep_ep/include/deep_ep/impls/ep/combine_reduce_epilogue.hpp", "// Load biases and rank partials", 157),
           source("deep_ep/include/deep_ep/impls/ep/combine_utils.hpp", "void combine_reduce_vf", 94)]
ENGRAM = [source("deep_ep/buffers/engram.py", "class EngramBuffer"),
          source("csrc/buffers/engram.hpp", "void write("),
          source("csrc/buffers/engram.hpp", "fetch(const torch::Tensor& indices"),
          source("deep_ep/include/deep_ep/impls/engram/engram_fetch.hpp", "void engram_fetch_impl", 140)]
BUCKET = source("csrc/buffers/bucket.hpp", "reduce_scatter(", 75)
LB = source("csrc/buffers/ep.hpp", "EventHandle lb_prefetch_weights", 86)
GATE = source("deep_ep/utils/gate.py", "def generate_topk_idx")


def classify(operator):
    op, category = operator["op"], operator["category"]
    refs = [API]
    if op == "AddFwdOp":
        return "partial", "Original combine's two BF16 biases implement alpha=1 addition; FP16/FP32 narrowing is forbidden. The apparent FP32 alternative lb_reduce_grads explicitly throws as unimplemented.", refs + COMBINE + [LB]
    if op == "CountNonzeroFwdOp":
        return "partial", "For bool inputs only, lossless bool->int64 makes expert IDs 0/1. Original native dispatch counts expert 1 once per reduction row, exactly count_nonzero. Numeric input cases require a nonzero predicate not present in DeepEP and are rejected.", refs + DISPATCH
    if op == "MoEPrePermuteFwdOp":
        return "supported_unverified", "Native byte-preserving expanded dispatch implements expert-major payload permutation, prefix sums and inverse slots for all seven cases; FP16 uses a lossless bit view.", refs + DISPATCH
    if op == "MoEPermuteAlignFwdOp":
        return "supported_unverified", "Flatten each route into a token; native expert_alignment computes aligned expert groups. Reformat native slot metadata and padded host counts into all three manifest outputs.", refs + DISPATCH
    if op == "PagedKVCacheGatherFwdOp":
        return "partial", "Engram write/fetch supports unscaled FP16/BF16 paged gathers after metadata address translation. FP8 dequantization is absent; that case is explicitly rejected.", refs + ENGRAM
    if op == "MoEPostPermuteFwdOp":
        return "unsupported", "The manifest requires sum_k(expert_output[inverse[t,k]] * topk_weights[t,k]). Native combine sums unweighted BF16 rows; weights are copied/restored, never multiplied into payloads. A Torch multiply or modified kernel would violate the adaptation boundary.", refs + COMBINE
    if op == "SumFwdOp":
        return "unsupported", "Every inventoried sum reduces 2048 or more elements, while native combine permits <=32 input routes per token and returns BF16. Recursive combine would insert BF16 rounding between partial sums, changing the reduction contract; bucket all_reduce/reduce_scatter are unimplemented.", refs + COMBINE + [BUCKET]
    if op in ("AllFwdOp", "AnyFwdOp"):
        return "unsupported", "Native expert counts can count bool values, but this op needs a boolean zero/nonzero or all-true predicate on that count. No original API returns that predicate; count->bool is a lossy conversion outside the allowed lossless-dtype/layout adaptation. General numeric inputs additionally need a nonzero predicate.", refs + DISPATCH
    if category == "elementwise":
        reason = "Own native payload arithmetic is unweighted BF16 row addition. No original API implements this op's unary/binary/selection/activation/generation contract; dispatch and fetch only copy bytes. Bit reinterpretation cannot replace arithmetic."
        refs += COMBINE
    elif category == "reduction":
        reason = "Own native combine implements bounded unweighted BF16 route sums, not this op's axis/scan/arg-index/statistics/normalization contract. Bucket reduction APIs explicitly throw; dependency test kernels and Torch utilities are excluded."
        refs += COMBINE + [BUCKET]
    elif category == "normalization":
        reason = "No original DeepEP kernel computes moments, reciprocal square roots, affine normalization, normalization gradients or adaptive/fused normalization outputs. Combine's optional bias is only addition."
        refs += COMBINE
    elif category == "rope":
        reason = "No original DeepEP API/kernel computes sinusoidal frequencies or pairwise rotary multiplication/addition; layout conversion and byte transport cannot implement RoPE."
        refs += DISPATCH
    elif category == "conv_pool":
        reason = "No original DeepEP API/kernel has convolution weights, sliding windows or pool reductions/indices. Engram is a storage-fetch buffer, not a convolution implementation."
        refs += ENGRAM
    elif category == "gemm":
        reason = "No own production matrix multiplication or quantized dot-product kernel/API. third-party/clangd-ascend/tests/bf16_gemm.asc is a toolchain test, not a DeepEP implementation."
        refs += [source("csrc/indexing/main.cpp", "// Kernel implements", 13)]
    elif category == "quantization":
        reason = "FP8 dispatch/fetch accept already quantized payloads and supplied scale-factor packs and copy them; they compute neither quantizer scales, conversion/clamping nor dequantization products."
        refs += [source("csrc/buffers/ep.hpp", "x.scalar_type() == torch::kFloat8_e4m3fn", 162)] + ENGRAM
    elif category == "attention":
        reason = "DeepEP transport/fetch APIs do not compute attention scores, softmax, top-k score selection, attention gradients/state merging or cache-write scatter. This manifest op needs those computations, beyond transport layout adaptation."
        refs += ENGRAM + [GATE]
    elif category == "moe":
        reason = "Dispatch/combine provide routing transport and unweighted reduction only. This op requires router scoring/selection or expert matrix products/activation absent from DeepEP's own kernels; gate.py generates test routing using Torch and is excluded."
        refs += COMBINE + [GATE]
    elif category == "sampling":
        reason = "No own probability filtering/ranking/CDF/sampling kernel. gate.py is Torch-based test traffic generation, not this manifest's logits/probability operator."
        refs += [GATE]
    elif category == "linear_attention":
        reason = "No recurrent/chunked linear-attention state update, matrix contraction or backward kernel. Expert-parallel communication does not implement the manifest state/output equations."
    elif category == "ssm":
        reason = "No state-space scan, state transition, chunk coupling or recurrent state kernel; communication counters/prefixes operate on route counts, not these model tensors."
        refs += DISPATCH
    elif op.startswith("Engram"):
        reason = "EngramBuffer only writes/fetches memory rows. This Engram manifest op includes projections, gated normalization/convolution or their backward/state update; shared naming is not mathematical compatibility."
        refs += ENGRAM
    elif op.startswith("MHC"):
        reason = "No own MHC projection, normalization, gating or weighted hyper-connection mixing. Unweighted BF16 communication reduction cannot supply the required outputs."
        refs += COMBINE
    else:
        reason = "No FFT/butterfly/Hadamard transform API or kernel in the complete own production source surface; byte-copy and unweighted route addition do not implement this transform."
    return "unsupported", f"{op}: {reason}", refs


def static_entries():
    root = WORKSPACE / "TileOPs/benchmarks/_cases"
    tree = ast.parse((root / "__init__.py").read_text())
    families = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_FAMILIES" for t in n.targets))
    entries = {}
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith("Op"):
                entries[node.value] = {"file": relative(path), "line": node.lineno,
                                       "sha256": digest(path), "family_wired": path.stem in families}
    return entries


def main():
    inventory = json.loads(INVENTORY.read_text())
    assert len(inventory["operators"]) == 205 and len(inventory["cases"]) >= 1982
    inventory_case_count = len(inventory["cases"])
    for path, sha in inventory["manifest_sha256"].items():
        assert digest(WORKSPACE / path) == sha, f"Manifest changed: {path}"
    assert len({(c["op"], c["case_id"]) for c in inventory["cases"]}) == inventory_case_count
    parsed = []
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text())
        parsed.append(relative(path))
    for path in vendor_sources():
        if path.suffix == ".py":
            ast.parse(path.read_text())
    # Exercise only CPU metadata formatting, never a tensor/operator reference.
    assert block_owners([0, 64, 128, 0], 64, 5, 4) == ([1, 2, 2, 4, 4], 192)
    assert paged_indices([[2, 0], [1, 2]], [0, 3, 5], [1, 0], 2, 3, 5) == [5, 0, 1, 2, 3]
    for bad in ([1, 0], [-64, 0]):
        try:
            block_owners(bad, 64, 2, 2)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Unaligned/negative host counts were accepted")
    deployment = json.loads((HERE / "deployment.json").read_text())
    log = (WORKSPACE / deployment["log_path"]).read_text()
    fatal = next((s for s in log.splitlines() if "fatal error:" in s), log[:300])
    build_blocked = not (WORKSPACE / ".cache/all-pairs/deepep/host/lib/build_receipt.json").is_file()
    assert build_blocked, "Audit must be updated explicitly after a successful native deployment"
    entries = static_entries()
    source_records, cases, operators = {}, [], []
    accepted = []
    by_op = defaultdict(list)
    for item in inventory["operators"]:
        semantic, reason, refs = classify(item)
        src = manifest_source(item)
        if item["op"] in PROVIDERS:
            fn = PROVIDERS[item["op"]].__name__
            refs += [source(HERE / "adapters.py", f"def {fn}(case):"),
                     source(HERE / "contracts.py", "def plan(op, params):")]
        source_records[item["op"]] = (semantic, reason, refs + [src])
    for case in inventory["cases"]:
        op = case["op"]
        semantic, reason, refs = source_records[op]
        status, adapter = "unsupported", None
        if op in PROVIDERS:
            try:
                plan(op, case["params"])
            except ValueError as exc:
                semantic, reason = "unsupported", str(exc)
            else:
                status, semantic = "deployment_blocked", "supported_unverified"
                adapter = ADAPTER
                accepted.append(case)
                reason = "Source-admitted implementation; native host deployment blocked: " + fatal
        row = {**case, "status": status, "semantic_status": semantic, "reason": reason,
               "adapter_path": adapter, "factory": PROVIDERS[op].__name__ if adapter else None,
               "sources": refs, "numerical_receipt_path": None, "input_protection_receipt_path": None,
               "native_call_attribution_receipt_path": None, "device_validation": "not_run_forbidden",
               "static_receipt_path": relative(HERE / "static_checks.json")}
        cases.append(row)
        by_op[op].append(row)
    # Check valid factories fail on real missing artifacts before reading inputs
    # or attempting to import any device module. Unsupported factories must fail
    # admission first. No native method or tensor operation is called here.
    class CaseMetadataOnly:
        def __init__(self, row):
            self.params = row["params"]

        @property
        def inputs(self):
            raise AssertionError("Static deployment preflight materialized workload inputs")

    for case in accepted:
        try:
            PROVIDERS[case["op"]](CaseMetadataOnly(case))
        except RuntimeError as exc:
            assert "deployment_blocked" in str(exc)
        else:
            raise AssertionError("Missing native build unexpectedly accepted")
    from benchmarks.ascend_baselines import UnsupportedCase
    rejected_factory_cases = 0
    for row in cases:
        if row["op"] not in PROVIDERS or row["status"] != "unsupported":
            continue
        try:
            PROVIDERS[row["op"]](CaseMetadataOnly(row))
        except UnsupportedCase:
            rejected_factory_cases += 1
        else:
            raise AssertionError("An unsupported dtype bypassed factory admission")
    for item in inventory["operators"]:
        op = item["op"]
        semantic, reason, refs = source_records[op]
        count = Counter(row["status"] for row in by_op[op])
        factory = entries.get(op)
        status = ("inventory_case_gap" if not by_op[op] and factory and factory["family_wired"]
                  else "workload_missing" if not by_op[op]
                  else "deployment_blocked" if count["deployment_blocked"] else "unsupported")
        operators.append({**item, "status": status, "semantic_status": semantic, "reason": reason,
                          "adapter_path": ADAPTER if op in PROVIDERS else None,
                          "factory": PROVIDERS[op].__name__ if op in PROVIDERS else None,
                          "workload_factory_source": factory, "sources": refs,
                          "case_status_counts": dict(count), "numerical_receipt_path": None,
                          "device_validation": "not_run_forbidden"})
    assert Counter(c["op"] for c in accepted) == {
        "AddFwdOp": 4, "CountNonzeroFwdOp": 1, "MoEPrePermuteFwdOp": 7,
        "MoEPermuteAlignFwdOp": 12, "PagedKVCacheGatherFwdOp": 2}
    assert not guard.attempts and not any(n.split(".")[0] in FORBIDDEN for n in sys.modules)
    schema = {"schema": "deepep-all-pairs-coverage-v1",
              "status_definitions": {
                  "supported_unverified": "Implemented and source-admitted; device result unknown.",
                  "deployment_blocked": "Implemented/source-admitted but native deployment prerequisites fail; not a numerical failure.",
                  "unsupported": "Inspected own source cannot meet this contract within allowed adaptation; no fallback registered.",
                  "workload_missing": "Frozen inventory contains no cases and no wired factory is visible; source-only conclusion retained.",
                  "inventory_case_gap": "A factory is now wired but the frozen inventory still has zero cases; root must recollect.",
                  "numerical_failed": "Reserved for actual device/oracle mismatch receipts; zero in this CPU task.",
                  "verified": "Reserved for full manifest outputs, input protection and native-call attribution; zero in this CPU task."},
              "structure": f"operators[205] plus cases[{inventory_case_count}], exact inventory (op,case_id,params); source file/line/line_end/SHA256 on every row; null device receipt paths mean no execution.",
              "precedence": "status is operational; semantic_status is independent. Add, CountNonzero and PagedKVCacheGather have partial dtype support. Missing-case ops do not create synthetic case rows."}
    snapshot = source_snapshot()
    catalog = {"schema": "deepep-source-catalog-v1", "implementation_and_build_sources": snapshot,
               "production_exports": source("csrc/python_api.cpp", "PYBIND11_MODULE", 74),
               "exclusions": ["deep_ep/utils/refs.py is a Torch oracle", "deep_ep/utils/gate.py generates random test traffic using Torch",
                              "third-party/deep_jit/tests and third-party/clangd-ascend/tests belong to dependencies, not DeepEP own kernels"],
               "key_findings": {"dispatch_is_byte_copy": DISPATCH, "combine_weights_are_not_payload_multipliers": COMBINE,
                                "engram_is_row_fetch_not_gated_convolution": ENGRAM, "bucket_reductions_unimplemented": [BUCKET],
                                "lb_prefetch_and_fp32_grad_reduction_unimplemented": [LB]}}
    check = {"schema": "deepep-static-checks-v1", "result": "pass", "device_execution": False,
             "operator_count": len(operators), "case_count": len(cases), "manifest_hashes_match": True,
             "providers": sorted(PROVIDERS), "source_admitted_cases": len(accepted),
             "adapter_syntax_files": parsed, "vendor_python_ast": "pass", "lazy_import_guard": "pass",
             "forbidden_import_attempts": guard.attempts, "deployment_preflight_before_inputs_cases": len(accepted),
             "unsupported_factory_cases_rejected_before_runtime": rejected_factory_cases,
             "host_metadata_checks": "aligned block owners, empty experts, invalid host counts, cross-page sequence starts",
             "native_host_syntax": deployment["status"], "native_host_syntax_log": deployment["log_path"],
             "numerical_verification": "not performed; no torch, tensors or native runtime imported"}
    summary = {"operators": len(operators), "cases": len(cases), "registered_providers": len(PROVIDERS),
               "operator_status_counts": dict(Counter(o["status"] for o in operators)),
               "case_status_counts": dict(Counter(c["status"] for c in cases)),
               "source_admitted_cases": len(accepted), "verified_cases": 0, "numerical_failed_cases": 0}
    # Validate every reference before publishing a coverage document. A source
    # span outside its file is not acceptable evidence, even if its hash exists.
    unique_sources = {(s["file"], s["line"], s["line_end"]): s
                      for row in operators + cases for s in row["sources"]}
    for (path, begin, end), ref in unique_sources.items():
        file = WORKSPACE / path
        assert 1 <= begin <= end <= len(file.read_text().splitlines()), path
        assert digest(file) == ref["sha256"], path
    check["validated_source_spans"] = len(unique_sources)
    coverage = {**schema, "library": "DeepEP-Ascend", "owner": "allpairs-remote/deepep",
                "inventory": {"file": relative(INVENTORY), "sha256": digest(INVENTORY)},
                "environment": {"architecture": deployment["architecture"], "cann": deployment["cann_path"],
                                "policy": "strict CPU-only; no NPU initialization/import/run/test/benchmark/profile"},
                "summary": summary, "deployment": {"status": deployment["status"], "reason": fatal,
                                                  "receipt_path": relative(HERE / "deployment.json")},
                "operators": operators, "cases": cases}
    missing = [{"op": o["op"], "status": o["status"], "factory_source": o["workload_factory_source"],
                "adapter_path": o["adapter_path"]} for o in operators if not o["inventoried_case_count"]]
    progress = {"schema": "deepep-progress-v1", "owner": "allpairs-remote/deepep", "worker_id": "bff46a18591c4c61",
                "updated_at": datetime.now(timezone.utc).isoformat(), "phase": "cpu_implementation_and_audit_complete",
                "device_policy": "CPU-only; zero runtime/device imports or execution", "summary": summary,
                "registered_providers": sorted(PROVIDERS), "remaining_issues": [
                    "Native host translation unit cannot compile: " + fatal,
                    "No source-built ARM64 extension or device/JIT validation; never reuse x86 verified receipts.",
                    "Shared root inventory expanded during this task from 1982 to " + str(inventory_case_count) + " cases; all current rows are covered.",
                    "No manifest/oracle change; no device evidence generated."], "missing_inventory_cases": missing,
                "coverage_path": relative(HERE / "coverage.json"), "checks_path": relative(HERE / "static_checks.json")}
    for name, value in (("coverage_schema.json", schema), ("source_catalog.json", catalog),
                        ("static_checks.json", check), ("coverage.json", coverage), ("progress.json", progress)):
        (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    # Versioned snapshot preserves evidence when this audit is rerun.
    run_dir = EVIDENCE / datetime.now(timezone.utc).strftime("cpu-audit-%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir()
    for name in ("coverage.json", "progress.json", "static_checks.json", "deployment.json", "source_catalog.json"):
        (run_dir / name).write_bytes((HERE / name).read_bytes())
    (run_dir / "native_host_syntax.log").write_text(log)
    print(json.dumps({"summary": summary, "evidence": relative(run_dir), "forbidden_imports": guard.attempts}, indent=2))


if __name__ == "__main__":
    main()
