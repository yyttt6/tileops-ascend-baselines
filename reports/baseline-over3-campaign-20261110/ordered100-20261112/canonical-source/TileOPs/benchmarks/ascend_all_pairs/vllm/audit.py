"""CPU-only, stdlib-only source/registration audit and coverage receipt generator.

Never invokes a Candidate factory, imports a tensor runtime, or compiles a kernel.
Run from any directory: PYTHONDONTWRITEBYTECODE=1 python3 <this-file>.
"""
import ast
from collections import Counter
import datetime
import hashlib
import importlib.abc
import inspect
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VENDOR = ROOT / "ascend_baselines/vllm-ascend"
INVENTORY = ROOT / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/inventory.json"
EVIDENCE = ROOT / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/vllm"
FORBIDDEN = {"torch", "torch_npu", "triton", "vllm", "vllm_ascend", "acl", "torchair", "tilelang"}


class DeviceImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise AssertionError("CPU audit attempted forbidden import: " + fullname)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, symbol=None):
    text = path.read_text()
    line = 1
    if symbol:
        found = re.search(r"^(?:async )?def " + re.escape(symbol) + r"\b", text, re.M)
        found = found or re.search(re.escape(symbol), text)
        if found:
            line = text[:found.start()].count("\n") + 1
    return {"file": str(path.relative_to(ROOT)), "line": line, "sha256": digest(path), "symbol": symbol}


T = "vllm_ascend/ops/triton/"
ROUTES = {
    "paged_append": [("csrc/torch_binding.cpp", "npu_sparse_flash_attention"), (T+"kda_state_copy_kernel.py", "_kda_state_copy_kernel"), (T+"rope.py", "rope_forward_triton_siso")],
    "fused_qk_norm_rope": [(T+"batch_invariant/rmsnorm.py", "rms_norm"), (T+"rope.py", "rope_forward_triton_siso")],
    "lse_attention": [("csrc/torch_binding.cpp", "npu_sparse_flash_mla"), ("csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp", None)],
    "nsa_selected": [("csrc/torch_binding.cpp", "npu_sparse_flash_attention")],
    "arithmetic": [(T+"muls_add.py", "muls_add_triton")],
    "mean": [(T+"batch_invariant/mean.py", "mean_dim")],
    "mean_pooling": [(T+"batch_invariant/mean.py", "mean_dim")],
    "sum_reduction": [(T+"batch_invariant/mean.py", "mean_dim"), (T+"muls_add.py", "muls_add_triton")],
    "average_pool": [(T+"batch_invariant/mean.py", "mean_dim"), (T+"muls_add.py", "muls_add_triton")],
    "normalization": [(T+"batch_invariant/rmsnorm.py", "rms_norm"), (T+"layernorm_gated.py", "layer_norm_fwd_npu"), (T+"muls_add.py", "muls_add_triton")],
    "adaptive_layer_norm": [(T+"layernorm_gated.py", "layer_norm_fwd_npu")],
    "group_norm": [(T+"layernorm_gated.py", "layer_norm_fwd_npu")],
    "gemm": [(T+"batch_invariant/matmul.py", "matmul_persistent")],
    "grouped_gemm": [(T+"batch_invariant/matmul.py", "matmul_persistent")],
    "pointwise_conv": [(T+"batch_invariant/matmul.py", "matmul_persistent")],
    "cumsum": [(T+"kda/cumsum.py", "chunk_local_cumsum_vector")],
    "activation": [(T+"activation/swiglustep.py", "swiglustep_forward_triton"), (T+"fused_gdn_gating.py", "fused_gdn_gating_patch"), (T+"muls_add.py", "muls_add_triton")],
    "expert": [(T+"batch_invariant/matmul.py", "matmul_persistent"), (T+"activation/swiglustep.py", "swiglustep_forward_triton")],
    "linear_attention": [(T+"kda/kda.py", "chunk_kda"), (T+"kda/kda.py", "fused_recurrent_kda"), (T+"fused_gdn_gating.py", "fused_gdn_gating_patch")],
    "rope": [(T+"rope.py", "rope_forward_triton_siso")],
    "amax": [("csrc/torch_binding.cpp", "npu_sparse_flash_attention")],
    "fused_topk": [("csrc/torch_binding.cpp", "moe_gating_top_k"), ("csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_tiling_arch35.cpp", "renorm_ != RENORM_NO")],
    "softmax": [("csrc/torch_binding.cpp", "moe_gating_top_k"), ("csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_tiling_arch35.cpp", "renorm_ != RENORM_NO")],
    "mhc_post": [("csrc/torch_binding.cpp", "npu_hc_post")],
    "dense_attention": [("csrc/torch_binding.cpp", "npu_sparse_flash_attention"), (T+"rope.py", "rope_forward_triton_siso")],
    "packed_attention": [("csrc/torch_binding.cpp", "npu_sparse_flash_attention"), (T+"rope.py", "rope_forward_triton_siso")],
    "mla_decode": [("csrc/torch_binding.cpp", "npu_sparse_flash_attention")],
    "dsa_decode": [("csrc/torch_binding.cpp", "npu_sparse_flash_attention")],
    "merge_states": [(T+"dcp/dcp_a2a.py", "_fused_dcp_lse_combine_kernel"), (T+"muls_add.py", "muls_add_triton")],
    "cache_transfer": [(T+"kda_state_copy_kernel.py", "_kda_state_copy_kernel"), ("vllm_ascend/ops/kda_state_copy_plan.py", "DEFAULT_KDA_BLOCK_SIZE"), (T+"muls_add.py", "muls_add_triton"), (T+"rope.py", "rope_forward_triton_siso")],
    "causal_conv": [("csrc/torch_binding.cpp", "npu_causal_conv1d_310"), ("csrc/moe/causal_conv1d_v310/op_host/causal_conv1d_v310_def.cpp", "ascend310p")],
}
NATIVE_DIRS = {
    "paged_append": "csrc/attention/sparse_flash_attention",
    "lse_attention": "csrc/attention/sparse_flash_mla", "nsa_selected": "csrc/attention/sparse_flash_attention",
    "amax": "csrc/attention/sparse_flash_attention", "dense_attention": "csrc/attention/sparse_flash_attention",
    "packed_attention": "csrc/attention/sparse_flash_attention", "mla_decode": "csrc/attention/sparse_flash_attention",
    "dsa_decode": "csrc/attention/sparse_flash_attention", "fused_topk": "csrc/moe/moe_gating_top_k",
    "softmax": "csrc/moe/moe_gating_top_k", "mhc_post": "csrc/moe/hc_post",
    "causal_conv": "csrc/moe/causal_conv1d_v310",
}


def audit():
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, DeviceImportBlocker())
    sys.path.insert(0, str(ROOT / "TileOPs"))
    from benchmarks.ascend_all_pairs.vllm import PROVIDERS
    from benchmarks.ascend_all_pairs.vllm.contracts import REVIEWS, decide
    from benchmarks.ascend_all_pairs.vllm.runtime import rewrite_imports
    inventory_bytes = INVENTORY.read_bytes()
    inventory = json.loads(inventory_bytes)
    (HERE / "inventory_snapshot.json").write_bytes(inventory_bytes)
    checks = []
    own_files = sorted(HERE.glob("*.py"))
    for path in own_files:
        compile(path.read_text(), str(path), "exec")
    checks.append({"check": "Python syntax compilation in memory", "files": len(own_files), "pass": True})
    assert not FORBIDDEN.intersection(sys.modules)
    checks.append({"check": "Registration import with hard device-runtime import blocker", "providers": len(PROVIDERS), "pass": True})
    names = {o["op"] for o in inventory["operators"]}
    assert len(names) == 205
    assert PROVIDERS.keys().isdisjoint(REVIEWS)
    assert PROVIDERS.keys() | REVIEWS.keys() == names
    assert all(callable(f) and len(inspect.signature(f).parameters) == 1 for f in PROVIDERS.values())
    checks.append({"check": "Exact 205-op partition and factory(case) registration", "pass": True})
    # Source-only API parameter verification: no native/vendor code executes.
    definitions = {}
    for items in ROUTES.values():
        for file, symbol in items:
            path = VENDOR / file
            assert path.is_file(), str(path)
            if path.suffix == ".py":
                for node in ast.parse(path.read_text()).body:
                    if isinstance(node, ast.FunctionDef):
                        definitions[node.name] = node
    api_calls = 0
    for path in own_files:
        if path.name == "audit.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr not in definitions:
                continue
            definition = definitions[node.func.attr]
            args = definition.args
            allowed = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
            for kw in node.keywords:
                assert kw.arg is None or kw.arg in allowed or args.kwarg, (path.name, node.lineno, kw.arg)
            if not any(isinstance(a, ast.Starred) for a in node.args) and not args.vararg:
                assert len(node.args) <= len(args.posonlyargs)+len(args.args), (path.name, node.lineno)
            api_calls += 1
    checks.append({"check": "Original Python API positional/keyword argument checks", "calls": api_calls, "pass": True})
    # Confirm every deployment import rewrite preserves every other AST node.
    sources = {VENDOR / f for v in ROUTES.values() for f, _ in v if f.endswith(".py")}
    sources |= set((VENDOR / (T+"kda")).glob("*.py"))
    sources |= {VENDOR / (T+"triton_utils.py"), VENDOR / "vllm_ascend/envs.py"}
    rewritten = 0
    for path in sorted(sources):
        original = ast.parse(path.read_text())
        imports = {a.name for n in ast.walk(original) if isinstance(n, ast.ImportFrom) for a in n.names}
        changed = rewrite_imports(ast.parse(path.read_text()), {x: None for x in imports}, "__audit_bindings")
        left, right = list(ast.walk(original)), list(ast.walk(changed))
        assert len(left) == len(right)
        for a, b in zip(left, right):
            if isinstance(a, ast.ImportFrom) and b.module == "__audit_bindings":
                b.module = a.module
        assert ast.dump(original) == ast.dump(changed), str(path)
        rewritten += 1
    selected = definitions["_fused_dcp_lse_combine_kernel"]
    restored = ast.parse(ast.unparse(ast.Module(body=[selected], type_ignores=[]))).body[0]
    assert ast.dump(selected) == ast.dump(restored)
    checks.append({"check": "Import-only AST rewrite and selected complete JIT definition round-trip", "files": rewritten, "pass": True})
    catalog = []
    for subtree in (VENDOR / "vllm_ascend/ops", VENDOR / "csrc"):
        for path in sorted(subtree.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".cpp", ".h", ".cuh"}:
                continue
            if any(part in {"build_cache", "third_party", "stub", "external"} for part in path.parts):
                continue
            entry = record(path)
            if path.suffix == ".py":
                entry["definitions"] = [{"name": n.name, "line": n.lineno} for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            catalog.append(entry)
    catalog_path = HERE / "source_catalog.json"
    catalog_path.write_text(json.dumps({"schema": "vllm-owned-source-catalog/v1", "excluded": ["build_cache", "third_party", "stub", "external"], "files": catalog}, indent=2)+"\n")
    runtime_site = Path("/opt/tileops-ci/runtime/lib/python3.12/site-packages")
    triton_missing = not (runtime_site / "triton").exists()
    deployment = {"runtime_site": str(runtime_site), "triton_package_present": not triton_missing,
                  "torch_package_present": (runtime_site / "torch").exists(),
                  "torch_npu_package_present": (runtime_site / "torch_npu").exists(),
                  "method": "Read-only filesystem inspection; no package imported. Runtime venv excludes system site-packages; no Triton editable .pth is present."}
    checks.append({"check": "Deployment package presence inspected without imports", "pass": True, "observation": deployment})
    evidence, operators, cases = {}, [], []
    for operator in inventory["operators"]:
        name = operator["op"]
        factory = PROVIDERS.get(name)
        if factory:
            src = [record(VENDOR / file, symbol) for file, symbol in ROUTES[factory.__name__]]
            if factory.__name__ in NATIVE_DIRS:
                src += [record(p) for p in sorted((VENDOR / NATIVE_DIRS[factory.__name__]).rglob("*")) if p.is_file() and p.suffix in {".h", ".cpp"}]
            adapter = record(Path(inspect.getsourcefile(factory)), factory.__name__)
            src += [adapter, record(HERE / "runtime.py"), record(HERE / "contracts.py")]
            apis = [symbol for _, symbol in ROUTES[factory.__name__]]
        else:
            src = [record(VENDOR / file) for file in REVIEWS[name]["evidence"]]
            adapter, apis = None, []
        manifest = ROOT / "TileOPs" / operator["manifest"]
        src += [record(manifest, name+":")]
        evidence[name] = (src, adapter, apis)
    for row in inventory["cases"]:
        name = row["op"]
        status, reason = decide(name, row.get("tensors", {}) | row["params"])
        assert status != "verified"
        if status in {"supported_unverified", "deployment_blocked"}:
            assert name in PROVIDERS
        semantic_status = "supported_unverified" if status in {"supported_unverified", "deployment_blocked"} else status
        semantic_reason = reason
        if status == "supported_unverified" and triton_missing:
            status = "deployment_blocked"
            reason = "The supplied /opt/tileops-ci/runtime Python3.12 environment has no Triton package or editable Triton path. Original Triton Ascend runtime/backend must be deployed in an authorized independent environment. No import or build was attempted."
        src, adapter, apis = evidence[name]
        cases.append({"op": name, "case_id": row["case_id"], "status": status,
                      "reason": reason, "semantic_status": semantic_status, "semantic_reason": semantic_reason, "adapter": adapter, "apis": apis,
                      "source_evidence": src, "numerical_receipt": None,
                      "device_validated": False, "outputs_verified": False,
                      "input_protection_verified": False, "native_launch_attribution_verified": False,
                      "workload_factory": "missing_in_shared_inventory" if name in inventory["missing_factory_ops"] else "available_in_shared_inventory"})
    for original in inventory["operators"]:
        name = original["op"]
        rows = [row for row in cases if row["op"] == name]
        counts = dict(Counter(row["status"] for row in rows))
        src, adapter, apis = evidence[name]
        operators.append({"op": name, "category": original["category"], "manifest_cases": len(rows),
                          "status": next(iter(counts)) if len(counts) == 1 else "mixed",
                          "status_counts": counts, "semantic_status_counts": dict(Counter(row["semantic_status"] for row in rows)), "registered": name in PROVIDERS, "adapter": adapter,
                          "apis": apis, "source_evidence": src,
                          "numerical_receipt": None,
                          "reason": (rows[0]["reason"] if len(counts) == 1 else "Case-specific metadata constraints; see all case rows."),
                          "scope_note": "Unsupported means no matching complete owned API in the inspected snapshot under allowed transformations; it is not a universal impossibility claim."})
    assert len(cases) == len(inventory["cases"])
    assert len({(x["op"], x["case_id"]) for x in cases}) == len(cases)
    checks.append({"check": "All inventory cases have explicit status, source evidence, adapter eligibility and empty numerical receipts", "cases": len(cases), "pass": True})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary = {"operators": len(operators), "registered_operators": len(PROVIDERS), "cases": len(cases),
               "case_status_counts": dict(Counter(x["status"] for x in cases)),
               "semantic_status_counts": dict(Counter(x["semantic_status"] for x in cases)),
               "device_runs": 0, "numerically_verified": 0,
               "missing_workload_factory_ops": inventory["missing_factory_ops"]}
    result = {"schema": "ascend-all-pairs-vllm-coverage/v1", "generated_at": now, "library": "vllm-ascend", "worker": "21ac4c053701495f",
              "status_definitions": {"supported_unverified": "Concrete original-library implementation and metadata binding; CPU source review only.",
                                     "deployment_blocked": "Concrete adapter exists, but the original ARM64 extension/OPP/target registration is absent or unresolved.",
                                     "unsupported": "Inspected owned API absent or mathematically/contractually incompatible under allowed transformations.",
                                     "adapter_blocked": "An owned source candidate exists, but a complete legal output/metadata binding is unresolved; no provider is registered.",
                                     "numerical_failed": "Reserved for a future complete device receipt; none produced here.",
                                     "verified": "Requires complete outputs, protected inputs and real owned launch attribution; none produced here."},
              "inventory": {"file": str(INVENTORY.relative_to(ROOT)), "sha256": hashlib.sha256(inventory_bytes).hexdigest(), "snapshot": "inventory_snapshot.json", "initial_requested_case_count": 1982, "frozen_case_count": len(cases)},
              "environment": {"task_policy": "strict CPU-only", "target": "aarch64 / CANN 9.1.0 (user supplied)", "model_constraint": "gpt-6-astra / xhigh requested; no delegation or model override", "historical_verified_reused": False},
              "summary": summary, "deployment": deployment, "source_catalog": {"file": "source_catalog.json", "sha256": digest(catalog_path)},
              "checks": checks, "operators": operators, "cases": cases}
    (HERE / "coverage.json").write_text(json.dumps(result, indent=2)+"\n")
    receipt = {"schema": "vllm-cpu-static-checks/v1", "at": now, "checks": checks, "summary": summary,
               "forbidden_imports_observed": [], "device_execution": False, "python_version": sys.version, "python_executable": sys.executable,
               "adapter_source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in own_files}}
    (HERE / "static_checks.json").write_text(json.dumps(receipt, indent=2)+"\n")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "cpu_static_checks.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    audit()
