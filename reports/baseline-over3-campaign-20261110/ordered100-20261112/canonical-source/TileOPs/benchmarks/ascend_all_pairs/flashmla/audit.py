"""Generate complete frozen-inventory coverage without importing any device runtime."""
import ast
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform

import yaml

from contract import ATTENTION, REDUCTION, POOL, ROUTES, classify, mutated_inputs
from native import WORKSPACE, VENDOR, RECEIPT, deployment_check, hashes, source_files

HERE = Path(__file__).resolve().parent
EVIDENCE = WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/flashmla"
INVENTORY = WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/inventory.json"


def reference(path, needle=None):
    path = WORKSPACE / path
    text = path.read_text()
    line = 1 if needle is None else next(i for i, value in enumerate(text.splitlines(), 1) if needle in value)
    return {"file": str(path.relative_to(WORKSPACE)), "line": line,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": str(value)}
    return value


def write(name, value):
    text = json.dumps(finite_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (HERE / name).write_text(text)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / name).write_text(text)


def factory_evidence():
    result = {}
    # AST-only discovery. No factory imports, workload instantiation or fake cases.
    for path in (WORKSPACE / "TileOPs/benchmarks/_cases").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith("Op"):
                result[node.value] = reference(path, node.value)
    return result


def exclusion(op, family):
    specific = {
        "GQABwdOp": "Dense backward is registered only under FLASH_MLA_IS_BUILD_ON_CUDA; Ascend exports no gradients.",
        "GemmFwdOp": "All inventoried GEMM contraction widths exceed native D512. The singleton max_logits dot-product route cannot split K without extra partial-sum math/narrowing.",
        "MLADecodeWithKVCacheFwdOp": "Manifest MLA score width is 512+64=576; both native Ascend prefill and decode require512, and the RoPE score tail cannot be cropped.",
        "MLAPagedFwdOp": "All declared workload rows use DK576, but native prefill/decode require512; BF16 layout changes cannot remove the64 score coordinates.",
        "DSAPagedFwdOp": "Manifest uses D576 and V3.2 656-byte FP8 rows; this checkout decodes D512 V4.1 528/288-byte formats only. Requantizing/recomputing scores is outside scope.",
        "MHADecodePagedWithKVCacheFwdOp": "Every inventoried MHA decode input is FP16; the native BF16-only prefill path cannot losslessly represent arbitrary FP16 values.",
        "NSATopKVarlenFwdOp": "Requires block-importance ranking and selected integer block IDs; FlashMLA consumes supplied indices and does not compute this selection.",
        "TopKSelectFwdOp": "Requires score ranking and top-k index output; FlashMLA consumes indices and returns no selected indices.",
        "FP8LightningIndexerFwdOp": "Requires per-head ReLU score aggregation, weights and top-k index selection; these are not exposed FlashMLA outputs.",
        "SoftmaxFwdOp": "Native API returns weighted values/max/LSE, not the probability tensor; constructing identity-valued attention is a new synthetic algorithm, outside the accepted adapter scope.",
        "LogSoftmaxFwdOp": "Native exposes LSE but not x-LSE; subtracting it would implement missing operator math outside this library API.",
        "AminFwdOp": "Exposed statistic is max, not min; negating inputs and outputs is additional numerical work beyond the accepted max-statistic layout route.",
        "MeanFwdOp": "Uniform native attention implements BF16-output means only; FP16/FP32 narrowing and FP32 output recovery are not lossless.",
        "AvgPool2dFwdOp": "FP16/FP32 cases cannot narrow losslessly; the BF16 case sets divisor_override=7 instead of native uniform-attention token count.",
        "AvgPool3dFwdOp": "FP16/FP32 cases cannot narrow losslessly; the BF16 case sets divisor_override=7 instead of native uniform-attention token count.",
        "MergeAttentionStatesFwdOp": "No standalone Ascend merge API accepts precomputed outputs and LSE. The CUDA combine kernel is not an Ascend provider.",
    }
    if op in specific:
        return specific[op]
    if "Indices" in op or op in {"ArgmaxFwdOp", "ArgminFwdOp"}:
        return "Requires exact arg-index output (including tie policy); FlashMLA returns value/max/LSE, not arg indices."
    if "CacheWrite" in op or op == "PagedKVCacheGatherFwdOp":
        return "FlashMLA has no standalone cache write/gather implementation; a torch-only layout copy is not library-owned operator execution."
    by_family = {
        "elementwise": "No standalone elementwise/residual output in the Ascend API; mathematical emulation with synthetic attention would introduce a new algorithm beyond layout adaptation.",
        "norm": "No Ascend normalization output or backward API; the fused norm/RoPE/attention entry explicitly rejects non-CUDA execution.",
        "rope": "The fused RoPE entry is CUDA-only; Ascend prefill expects already-encoded Q/K and exposes no rotation output.",
        "gemm": "The native singleton-QK statistic is limited to BF16 K<=512. These contracts require larger contractions or FP8/INT8/W4 dequantization/accumulation APIs absent from this route.",
        "quantization": "The decode kernel consumes its fixed FP8/FP4 cache format but exposes no quantization/dequantization tensor output or scale generation API.",
        "convolution": "No convolution API or convolution output; attention's softmax-weighted contraction is not the required unnormalized spatial contraction.",
        "pool": "Required pooling outputs or divisor semantics are not exposed by the accepted native max/uniform-mean route.",
        "reduction": "No matching exposed statistic: native returns only attention output, maximum and natural-log LSE, not this sum/product/scan/moment/norm/logical result.",
        "moe": "No expert routing, permutation, grouped expert GEMM or activation implementation in FlashMLA's Ascend API.",
        "sampling": "FlashMLA consumes selection metadata; it has no sampling, sorting, top-k/top-p probability-mask or random-draw output.",
        "linear_attention": "Normalized sparse softmax attention does not implement recurrent delta/gated linear attention, state updates or their gradients.",
        "mamba": "No state-space recurrence, selective scan, state passing or causal-convolution output in the Ascend API.",
        "fft": "No FFT API; normalized attention cannot implement the complex Fourier transform by permitted layout operations.",
        "transform": "No Hadamard transform API; sign-weighted unnormalized transforms are not native attention outputs.",
        "sequence_modeling": "No Engram or MHC gate/convolution/mixing API; these compound outputs cannot be selected from sparse attention.",
    }
    return by_family.get(family, "No equivalent Ascend API output for the manifest contract within permitted adaptations.")


def main():
    raw_inventory = INVENTORY.read_bytes()
    inventory = json.loads(raw_inventory)
    inventory_digest = hashlib.sha256(raw_inventory).hexdigest()
    assert len(inventory["operators"]) == 205 and len(inventory["cases"]) >= 1982
    assert len({(c["op"], c["case_id"]) for c in inventory["cases"]}) == len(inventory["cases"])
    snapshot = EVIDENCE / "inventory_snapshots" / (inventory_digest + ".json")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_bytes(raw_inventory)
    factories = factory_evidence()
    manifest = {}
    for op in inventory["operators"]:
        path = WORKSPACE / "TileOPs" / op["manifest"]
        if str(path) not in manifest:
            manifest[str(path)] = yaml.safe_load(path.read_text())
    vendor = "ascend_baselines/FlashMLA/"
    sources = {
        "public_api": reference(vendor + "csrc/api/api.cpp", "PYBIND11_MODULE"),
        "prefill": reference(vendor + "csrc/api/sparse_prefill.cpp", "static std::vector<at::Tensor> sparse_prefill_fwd"),
        "ascend_dispatch": reference(vendor + "csrc/api/sparse_prefill.cpp", "#ifdef FLASH_MLA_IS_BUILD_ON_ASCEND"),
        "restrictions": reference(vendor + "csrc/api/sparse_prefill.cpp", "TORCH_CHECK(d_qk == 512"),
        "statistics": reference(vendor + "csrc/ascend_kernels/prefill/sparse/kernel_body.h", "void Kernel<CONFIG>::get_final_lse_and_denorm_vf"),
        "weighted_mean": reference(vendor + "csrc/ascend_kernels/prefill/sparse/kernel_body.h", "asc_exp_sub(p[local_row][local_col]"),
        "native_shape": reference(vendor + "csrc/ascend_kernels/prefill/sparse/config.h", "static constexpr uint32_t D_QK"),
        "decode": reference(vendor + "csrc/api/sparse_decode.cpp", "TORCH_CHECK(d_qk == 512"),
        "cache_format": reference(vendor + "csrc/cuda_kernels/kv_cache_format.h", "BYTES_PER_TOKEN"),
        "cuda_only_backward": reference(vendor + "csrc/api/dense_bwd.cpp", "#ifdef FLASH_MLA_IS_BUILD_ON_CUDA"),
        "cuda_only_fused": reference(vendor + "csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp", "only supported on CUDA GPUs"),
    }
    deployed, detail = deployment_check()
    build = json.loads(RECEIPT.read_text()) if RECEIPT.is_file() else {}
    current_status = "supported_unverified" if deployed else "deployment_blocked"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    oracle_issues = []
    for row in inventory["cases"]:
        if row["op"] == "GQADenseFwdOp" and row["params"]["q"][0][1] != 1 and "q_scale" not in row["params"] and "rope_cos" not in row["params"]:
            oracle_issues.append({
                "op": row["op"], "case_id": row["case_id"], "status": "suspected_from_source_only",
                "reason": "Current _gqa_dense_workload chooses GQADenseDecodeCall without scales/RoPE. Its gen_inputs emits S_q=1 and decode reference ignores the multi-query causal mask; manifest declares S_q>1. Original oracle was not modified.",
                "manifest_q": row["params"]["q"],
                "sources": [reference("TileOPs/benchmarks/_cases/attention.py", "def _gqa_dense_workload"),
                            reference("TileOPs/workloads/attention/gqa/dense.py", "class GQADenseDecodeCall"),
                            reference("TileOPs/workloads/attention/gqa/dense.py", "q = torch.randn(self.batch, 1,")],
            })
    issue_keys = {(r["op"], r["case_id"]) for r in oracle_issues}
    rows, operators = [], []
    for op in inventory["operators"]:
        name = op["op"]
        file = WORKSPACE / "TileOPs" / op["manifest"]
        spec = manifest[str(file)][name]
        manifest_source = reference(file, name + ":")
        op_cases = []
        for case in (c for c in inventory["cases"] if c["op"] == name):
            supported, reason = classify(name, case["params"])
            if name not in ROUTES:
                reason = exclusion(name, op["family"])
            row = dict(case)
            row.update(library="FlashMLA", status=current_status if supported else "unsupported",
                       semantic_status="supported_unverified" if supported else "unsupported",
                       reason=reason, deployment_reason=None if not supported or deployed else detail,
                       adapter=("TileOPs/benchmarks/ascend_all_pairs/flashmla/boolean.py" if name == "LogicalOrFwdOp" else "TileOPs/benchmarks/ascend_all_pairs/flashmla/adapters.py") if supported else None,
                       native_api="FlashMLA::sparse_prefill_fwd" if supported else None,
                       manifest_source=manifest_source, vendor_sources=list(sources.values()),
                       required_outputs=[*spec["signature"].get("outputs", {}),
                                         *mutated_inputs(spec["signature"], case["params"])],
                       numerical_receipt=None, full_output_check="not_run", input_preservation_check="not_run",
                       native_call_attribution=("source_pinned; runtime_not_executed; build_status=" + build.get("status", "not_built")) if supported else "not_applicable",
                       workload_factory=factories.get(name),
                       harness_status="factory_present_static" if name in factories else "missing_workload_factory",
                       oracle_issue="oracle_issues.json" if (name, case["case_id"]) in issue_keys else None)
            rows.append(row)
            op_cases.append(row)
        supported = any(c["semantic_status"] == "supported_unverified" for c in op_cases)
        # MLAVarlen has real declared BF16 rows but no shared workload factory.
        # Preserve the executable adapter without fabricating expanded cases.
        no_factory_route = name == "MLAVarlenFwdOp" and not op_cases
        supported = supported or no_factory_route
        operators.append({
            **op, "library": "FlashMLA", "status": current_status if supported else "unsupported",
            "semantic_status": "supported_unverified" if supported else "unsupported",
            "reason": ("Executable BF16 concatenated K_nope/K_pe sparse attention with full O and FP32 LSE; shared workload factory missing, no cases invented."
                       if no_factory_route else ("Per-case restrictions and verdicts are listed below." if supported else exclusion(name, op["family"]))),
            "adapter": ("TileOPs/benchmarks/ascend_all_pairs/flashmla/boolean.py" if name == "LogicalOrFwdOp" else "TileOPs/benchmarks/ascend_all_pairs/flashmla/adapters.py") if name in ROUTES else None,
            "registered": name in ROUTES, "case_status_counts": dict(Counter(c["status"] for c in op_cases)),
            "semantic_case_status_counts": dict(Counter(c["semantic_status"] for c in op_cases)),
            "manifest_source": manifest_source, "vendor_sources": list(sources.values()),
            "reference_api": spec.get("ref_api"), "manifest_workload_rows": len(spec.get("workloads", [])),
            "workload_factory": factories.get(name),
            "harness_status": ("missing_workload_factory" if name not in factories else "frozen_inventory_has_no_cases" if not op_cases else "factory_present_static"),
            "numerical_receipt": None,
        })
    summary = {"operators": len(operators), "cases": len(rows), "providers": len(ROUTES),
               "operator_status_counts": dict(Counter(o["status"] for o in operators)),
               "case_status_counts": dict(Counter(r["status"] for r in rows)),
               "semantic_case_status_counts": dict(Counter(r["semantic_status"] for r in rows)),
               "device_runs": 0, "verified_cases": 0, "numerical_failures": 0,
               "missing_workload_factories": [o["op"] for o in operators if o["workload_factory"] is None],
               "zero_case_inventory_ops": [o["op"] for o in operators if not o["inventoried_case_count"]],
               "suspected_oracle_cases": len(oracle_issues)}
    initial_missing = inventory.get("missing_factory_ops_initial", [])
    summary["original_1982_cases_retained"] = sum(r["op"] not in initial_missing for r in rows) if initial_missing else len(rows)
    summary["expanded_inventory_cases"] = len(rows) - 1982
    coverage = {"schema": "flashmla-all-pairs-coverage/v1", "generated_at": now,
                "schema_notes": {"status": "unsupported = no permitted route for inspected case; deployment_blocked = implemented semantic route with concrete compiler/deployment obstruction; supported_unverified = implemented route and source-pinned local build, no device validation",
                                 "semantic_status": "Independent of deployment. Always unverified for supported routes in this CPU-only task.",
                                 "numerical_receipt": "Null means no execution; static checks and historical x86 passes are never numerical receipts.",
                                 "nonfinite_float": "Tagged textual representation preserves nonfinite inventory params in strict JSON.",
                                 "scope": "All205 operators and every case of the source-pinned shared inventory. The parent expanded1982 to2062 during this task; the original1982 are retained. Missing factories are tracked independently of supplied manifest cases."},
                "library": "FlashMLA", "architecture": platform.machine(), "npu_allowed": False,
                "inventory": str(INVENTORY.relative_to(WORKSPACE)), "inventory_sha256": inventory_digest,
                "inventory_snapshot": str(snapshot.relative_to(WORKSPACE)),
                "summary": summary, "sources": sources, "source_sha256": hashes(source_files()),
                "adapter_sha256": hashes(list(HERE.glob("*.py"))), "build_receipt": str(RECEIPT.relative_to(WORKSPACE)),
                "build_status": build.get("status", "not_built"), "operators": operators, "cases": rows}
    write("coverage.json", coverage)
    write("oracle_issues.json", {"schema": "flashmla-oracle-review/v1", "device_runs": 0, "issues": oracle_issues})
    progress = {"schema": "flashmla-progress/v1", "worker": "55a9f63d207142c9", "owner": "allpairs-remote/flashmla",
                "phase": "implemented_cpu_audited", "updated_at": now, "npu_allowed": False,
                **summary, "build_status": build.get("status", "not_built"), "build_receipt": str(RECEIPT.relative_to(WORKSPACE)),
                "remaining": ["Run full manifest outputs, input preservation and real native-call attribution only in a separately authorized NPU environment.",
                              "Resolve shared workload factory/inventory gaps and independently review oracle_issues.json.",
                              "CPU build success is not proof of extension-load or NPU compatibility."],
                "coverage": "coverage.json", "oracle_issues": "oracle_issues.json"}
    write("progress.json", progress)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
