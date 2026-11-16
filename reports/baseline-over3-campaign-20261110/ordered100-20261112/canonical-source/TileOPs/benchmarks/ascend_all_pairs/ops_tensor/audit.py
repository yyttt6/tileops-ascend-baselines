"""Reproducible CPU-only source/registration/coverage audit. No tensor imports."""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.abc
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
EVIDENCE = WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_tensor"
SNAPSHOT = EVIDENCE / "inventory_snapshot.json"
sys.path.insert(0, str(WORKSPACE / "TileOPs"))


class ForbidDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "torch_npu", "acl", "aclnn", "triton", "tilelang"}:
            raise AssertionError(f"Device-capable import attempted in CPU audit: {fullname}")
        return None


sys.meta_path.insert(0, ForbidDeviceImports())
from benchmarks.ascend_all_pairs.ops_tensor import PROVIDERS
from benchmarks.ascend_all_pairs.ops_tensor.build import CACHE, SOURCES, VENDOR, build_spec, fingerprint
from benchmarks.ascend_all_pairs.ops_tensor.contract import IMPLEMENTED, convolution_plan, pooling_plan, support_reason
from benchmarks.ascend_all_pairs.ops_tensor.semantics import ANCHORS, GEMM, GROUPED, REDUCE, SWIGLU, unsupported_assessment


def relative(path):
    return str(Path(path).relative_to(WORKSPACE))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_ref(path, anchor=None):
    path = Path(path)
    if not path.is_absolute():
        path = VENDOR / path
    lines = path.read_text().splitlines()
    line = 1
    if anchor:
        hits = [i for i, text in enumerate(lines, 1) if anchor in text]
        if not hits:
            raise AssertionError(f"Missing source anchor {anchor!r}: {path}")
        line = hits[0]
    return dict(file=relative(path), line=line, sha256=sha(path), anchor=anchor)


def evidence_refs(files):
    return [source_ref(path, ANCHORS.get(path)) for path in dict.fromkeys(files)]


def workload_factories():
    # Parse only _FAMILIES and string literals; no importing benchmark workloads.
    directory = WORKSPACE / "TileOPs/benchmarks/_cases"
    root = ast.parse((directory / "__init__.py").read_text())
    families = next(ast.literal_eval(n.value) for n in root.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_FAMILIES" for t in n.targets))
    result = {}
    for family in families:
        path = directory / f"{family}.py"
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(r"\w+(?:Fwd|Bwd)Op", node.value):
                result[node.value] = dict(file=relative(path), line=node.lineno, sha256=sha(path))
    return result


def source_corpus():
    paths = sorted(p for folder in (VENDOR / "include", VENDOR / "examples")
                   for p in folder.rglob("*") if p.suffix in (".h", ".hpp", ".cpp", ".asc", ".c", ".cc"))
    entries = []
    for path in paths:
        text = path.read_text(errors="replace")
        global_lines = [dict(line=i, declaration=line.strip()) for i, line in enumerate(text.splitlines(), 1)
                        if "__global__" in line and not line.lstrip().startswith("//")]
        entries.append(dict(file=relative(path), sha256=sha(path), global_entry_lines=global_lines))
    # The supplied source manifest is explicitly authorized provenance, not a credential file.
    manifest_path = Path("/opt/tileops-all-pairs/control/source_manifest.json")
    original = json.loads(manifest_path.read_text())
    hashes = {entry["path"]: entry["sha256"] for entry in original["files"]
              if entry["path"].startswith("ascend_baselines/ops-tensor/")}
    mismatch = [e["file"] for e in entries if hashes.get(e["file"]) != e["sha256"]]
    assert not mismatch, f"Vendor sources differ from supplied snapshot: {mismatch}"
    corpus = dict(schema="ops-tensor-source-corpus-v1", files=entries, file_count=len(entries),
                  source_manifest=source_ref(manifest_path) if manifest_path.is_relative_to(WORKSPACE) else
                      dict(file=str(manifest_path), sha256=sha(manifest_path)),
                  source_snapshot_mismatches=mismatch,
                  public_elementwise_definitions=0,
                  scope="All production include/ and examples/ C/C++/ASC sources; no third-party library attribution.")
    decl_hits = []
    for path in paths:
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if "acltensorElementwiseBinaryExecute(" in line:
                decl_hits.append(dict(file=relative(path), line=i))
    assert len(decl_hits) == 1 and decl_hits[0]["file"].endswith("include/cann_ops_tensor.h")
    corpus["public_elementwise_occurrences"] = decl_hits
    write_json(HERE / "source_inventory.json", corpus)
    return corpus


def static_checks(inventory, factories):
    checks = []
    assert set(PROVIDERS) == set(IMPLEMENTED)
    assert all(callable(factory) for factory in PROVIDERS.values())
    checks.append(dict(check="lazy_import_and_registration", status="passed", providers=sorted(PROVIDERS),
                       device_module_imports=[], factory_invocations=0))
    for path in HERE.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
        compile(path.read_text(), str(path), "exec")  # in-memory syntax check only
    checks.append(dict(check="all_python_ast_and_in_memory_compile", status="passed"))
    for kind in SOURCES:
        path = HERE / "native" / f"{kind}.cpp"
        text = path.read_text()
        assert f'#include "{SOURCES[kind]}"' in text
        assert "__global__" not in text and "__aicore__" not in text and "__simd_vf__" not in text
        assert not re.search(r"\b(?:aclInit|aclrtSetDevice|aclrtCreateStream)\s*\(", text)
    checks.append(dict(check="native_bridge_no_new_kernel_or_device_initialization", status="passed",
                       detail="Includes complete original source; wraps existing launchers. Original demo main is renamed and never called."))
    banned = {"matmul", "mm", "bmm", "conv1d", "conv2d", "conv3d", "einsum", "sum", "amax", "amin", "max_pool1d",
              "max_pool2d", "max_pool3d", "trunc", "add", "relu", "softmax", "silu"}
    for module in ("adapters.py", "reductions.py"):
        calls = ast.parse((HERE / module).read_text())
        for node in ast.walk(calls):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in banned:
                raise AssertionError(f"Forbidden external math call in {module}, line {node.lineno}: {node.func.attr}")
    checks.append(dict(check="no_torch_or_dependency_target_computation", status="passed"))
    plans, supported = [], []
    for row in inventory["cases"]:
        op = row["op"]
        params = row.get("tensors", {}) | row["params"]
        if op in PROVIDERS and support_reason(op, params) is None:
            supported.append((op, row["case_id"]))
            if op.startswith("Conv"):
                plan = convolution_plan(params)
                rank = plan["rank"]
                # Independently reconstruct the unfold shape from padded spatial extents.
                out = tuple((s + 2 * p - d * (k - 1) - 1) // t + 1 for s, p, d, k, t in zip(
                    plan["spatial"], plan["padding"], plan["dilation"], plan["kernel"], plan["stride"]))
                assert out == plan["output"]
                permutation = (0, 1, *range(3, 3 + rank), 2, *range(3 + rank, 3 + 2 * rank))
                assert sorted(permutation) == list(range(3 + 2 * rank))
                plans.append(dict(op=op, case_id=row["case_id"], plan=plan))
            if op.startswith(("MaxPool", "AvgPool")) or op in ("AdaptiveMaxPool2dFwdOp", "AdaptiveAvgPool2dFwdOp"):
                plan = pooling_plan(op, params)
                if "MaxPool" in op:
                    assert plan["max_window"] <= 64
                plans.append(dict(op=op, case_id=row["case_id"], plan=plan))
            if op in ("SumFwdOp", "MeanFwdOp"):
                from math import prod
                shape = params["x"][0]
                dims = params.get("dim")
                dims = list(range(len(shape))) if dims is None else ([dims] if isinstance(dims, int) else dims)
                dims = [d % len(shape) for d in dims]
                output = [1 if i in dims else size for i, size in enumerate(shape)] if params.get("keepdim") else [size for i, size in enumerate(shape) if i not in dims]
                assert output == list(params["output"][0])
                plans.append(dict(op=op, case_id=row["case_id"], reduction_extent=prod(shape[d] for d in dims),
                                  output_shape=output, native_dtype="float32", coefficient="1/R" if op == "MeanFwdOp" else "1"))
    checks.append(dict(check="all_case_metadata_and_layout_plans", status="passed", eligible_case_count=len(supported),
                       layout_plans=plans, device_values_examined=False))
    assert len({(c["op"], c["case_id"]) for c in inventory["cases"]}) == len(inventory["cases"])
    assert len(inventory["operators"]) == 205
    checks.append(dict(check="inventory_key_integrity", status="passed", operators=205, cases=len(inventory["cases"])))
    return checks


def native_builds():
    result = {}
    for kind in SOURCES:
        spec = build_spec(kind)
        receipt = Path(spec["cache_directory"]) / "build.json"
        if not receipt.is_file():
            result[kind] = dict(status="deployment_blocked", reason="No build receipt for current source hash", expected_receipt=str(receipt))
            continue
        data = json.loads(receipt.read_text())
        if data["status"] == "compiled":
            binary = Path(data["binary"])
            assert fingerprint([binary]) == data["binary_sha256"]
            output = subprocess.check_output(["nm", "-D", "--defined-only", str(binary)], text=True)
            symbols = sorted({line.split()[-1] for line in output.splitlines() if line.split()})
            entry = "allpairs_opstensor_" + kind
            assert entry in symbols
            # Inspect the ELF only. Runtime dladdr/launch ownership remains unverified.
            machine = subprocess.check_output(["readelf", "-h", str(binary)], text=True)
            assert "AArch64" in machine if data["host_arch"] == "aarch64" else "X86-64" in machine
            data["static_symbol_ownership"] = dict(entry=entry, defined_in=relative(binary), checked_with="nm -D --defined-only")
            data["runtime_ownership_verified"] = False
        result[kind] = data | {"receipt": relative(receipt), "receipt_sha256": sha(receipt)}
    write_json(HERE / "build_receipts.json", result)
    return result


def main():
    inventory = json.loads(SNAPSHOT.read_text())
    ops = {o["op"]: o for o in inventory["operators"]}
    factories = workload_factories()
    corpus = source_corpus()
    checks = static_checks(inventory, factories)
    builds = native_builds()
    now = datetime.now(timezone.utc).isoformat()
    static_receipt = HERE / "static_checks.json"
    write_json(static_receipt, dict(schema="ops-tensor-static-checks-v1", updated_at=now, checks=checks,
               build_status={k: v["status"] for k, v in builds.items()},
               source_snapshot_files_unchanged=corpus["file_count"],
               npu_initializations=0, tensor_imports=0, device_runs=0, numerical_verifications=0))
    cases, operators = [], []
    for row in inventory["cases"]:
        op = row["op"]
        info = ops[op]
        params = row.get("tensors", {}) | row["params"]
        factory_missing = op not in factories
        reason = support_reason(op, params) if op in PROVIDERS else None
        kind = IMPLEMENTED.get(op)
        if kind and reason is None:
            sources = [SOURCES[kind]]
            status = "supported_unverified" if builds[kind]["status"] == "compiled" else "deployment_blocked"
            reason = ("Complete adapter implemented and source/metadata checked; original native source compiled for this host. "
                      "Device validation forbidden: full manifest outputs, input protection and actual launch ownership have NOT been tested.")
            if status == "deployment_blocked":
                reason = builds[kind].get("reason", "CPU compilation failed; inspect exact build receipt.")
            semantic_status = "supported"
        elif kind:
            sources = [SOURCES[kind]]
            if op == "MoEGroupedGemmFwdOp":
                sources.append(SWIGLU)
            status, semantic_status = "unsupported", "unsupported_within_allowed_adaptation"
        else:
            reason, sources = unsupported_assessment(op, row["family"], params)
            status, semantic_status = "unsupported", "unsupported_within_allowed_adaptation"
        record = dict(op=op, case_id=row["case_id"], library="ops-tensor", status=status,
                      semantic_status=semantic_status, reason=reason,
                      in_original_1982_scope=info.get("previous_factory_case_count", info["inventoried_case_count"]) > 0,
                      workload_factory_status="missing" if factory_missing else "present_static",
                      workload_factory_source=factories.get(op), source=evidence_refs(sources),
                      adapter_path=relative(Path(PROVIDERS[op].__code__.co_filename)) if kind else None,
                      factory=("benchmarks.ascend_all_pairs.ops_tensor:PROVIDERS[" + repr(op) + "]") if kind else None,
                      provider_registered=kind is not None, case_admitted=semantic_status == "supported",
                      api=("ops-tensor::" + ("LaunchKernel / Blaze::GemmUniversal" if kind == "gemm" else
                           "reduction_compute" if kind == "reduction" else str(kind))) if kind else None,
                      build_kind=kind, build_receipt=builds[kind].get("receipt") if kind else None,
                      static_receipt=relative(static_receipt), numerical_receipt=None,
                      full_manifest_outputs_validated=False, input_protection_validated=False,
                      runtime_ownership_validated=False, device_validated=False)
        if kind:
            fn = PROVIDERS[op]
            record["adapter_line"] = fn.__code__.co_firstlineno
            record["adapter_sha256"] = sha(Path(fn.__code__.co_filename))
        cases.append(record)
    for op, info in ops.items():
        rows = [r for r in cases if r["op"] == op]
        counts = dict(Counter(r["status"] for r in rows))
        status = "supported_unverified" if counts.get("supported_unverified") else (
            "deployment_blocked" if counts.get("deployment_blocked") else "unsupported")
        manifest = WORKSPACE / "TileOPs" / info["manifest"]
        operators.append(dict(op=op, family=info["family"], status=status, case_count=len(rows), case_status_counts=counts,
                              admitted_cases=sum(r["case_admitted"] for r in rows),
                              workload_factory_status="present_static" if op in factories else "missing",
                              source=rows[0]["source"] if rows else [],
                              reason=rows[0]["reason"] if rows else "No inventoried case; no test fabricated.",
                              adapter_path=relative(Path(PROVIDERS[op].__code__.co_filename)) if op in PROVIDERS else None,
                              manifest=source_ref(manifest, op + ":"), numerical_receipt=None))
    old_cases = [r for r in cases if r["in_original_1982_scope"]]
    assert len(old_cases) == 1982
    missing_ops = sorted(set(ops) - set(factories))
    summary = dict(operators=len(operators), cases=len(cases), providers=len(PROVIDERS),
                   operator_status_counts=dict(Counter(o["status"] for o in operators)),
                   case_status_counts=dict(Counter(c["status"] for c in cases)),
                   original_1982_case_status_counts=dict(Counter(c["status"] for c in old_cases)),
                   added_manifest_cases=len(cases) - len(old_cases),
                   missing_workload_factory_ops=missing_ops,
                   missing_workload_factory_case_count=sum(c["workload_factory_status"] == "missing" for c in cases),
                   verified=0, numerical_failed=0, device_attempts=0, deployment_blocked=sum(c["status"] == "deployment_blocked" for c in cases))
    coverage = dict(schema="ops-tensor-all-pairs-coverage-v1", updated_at=now, library="ops-tensor", worker="ede7adbc232a4469",
                    status_definitions={
                        "supported_unverified": "Concrete adapter, admitted metadata, CPU compilation complete; no numerical/device claim.",
                        "deployment_blocked": "Concrete equivalent adapter but exact current compile/dependency check failed; not mathematical unsupported.",
                        "unsupported": "No equivalent complete path within allowed adaptations, with nearest source and specific reason.",
                        "numerical_failed": "Reserved for actual future complete-output device failure; no such current runs.",
                        "verified": "Reserved for future full manifest output/input protection/native runtime ownership receipts; zero here."},
                    scope_note="Assigned 205 operators/1982 factory-backed cases; shared inventory expanded during work to 2062 manifest cases. Both scopes are explicit.",
                    inventory=dict(snapshot=relative(SNAPSHOT), sha256=sha(SNAPSHOT)), summary=summary,
                    environment=dict(host="aarch64", cann="9.1.0", npu_allowed=False, binaries_loaded=False,
                                     historical_x86_receipts_inherited=False),
                    static_checks=relative(static_receipt), build_receipts=relative(HERE / "build_receipts.json"),
                    source_corpus=relative(HERE / "source_inventory.json"),
                    oracle_observations=relative(HERE / "oracle_notes.json") if (HERE / "oracle_notes.json").exists() else None,
                    input_adaptation_policy="No input-dependent cached packing; current group metadata read in run; output buffers private; complete host work timed.",
                    operators=operators, cases=cases)
    write_json(HERE / "coverage.json", coverage)
    write_json(EVIDENCE / "coverage.json", coverage)
    progress = dict(schema="ops-tensor-progress-v1", updated_at=now, worker="ede7adbc232a4469", owner="allpairs-remote/ops_tensor",
                    phase="complete_cpu_only", summary=summary, npu_allowed=False, npu_initializations=0, device_executions=0,
                    completed=["All 205 operators source-screened with file/line/hash evidence", "Every current inventory case classified",
                               "Lazy providers and four unmodified native example bridges implemented", "CPU-only AArch64 compilation and ELF ownership inspection",
                               "Blocked-import registration check, AST audit, metadata/layout validation, original source snapshot hash comparison"],
                    issues=["Device numerical validation, input protection and runtime native attribution remain forbidden/unverified.",
                            "No latency or performance claim; per-window/original-chunk host launch counts can be very large.",
                            "Sum/mean/average pooling select the original GEMM with constant operands; floating accumulation/rounding still needs original manifest validation.",
                            "Shared inventory expanded from 1982 to 2062 cases during this task; frozen snapshot retained.",
                            f"Missing workload factories ({len(missing_ops)}): " + ", ".join(missing_ops)],
                    next_authorized_work="None in this CPU-only assignment. Parent can collect source/evidence; future device validation requires separate authorization and host guard.",
                    artifacts=[relative(HERE / f) for f in ("__init__.py", "coverage.json", "static_checks.json", "build_receipts.json", "source_inventory.json")])
    progress["oracle_observations"] = coverage["oracle_observations"]
    write_json(HERE / "progress.json", progress)
    write_json(EVIDENCE / "progress.json", progress)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
