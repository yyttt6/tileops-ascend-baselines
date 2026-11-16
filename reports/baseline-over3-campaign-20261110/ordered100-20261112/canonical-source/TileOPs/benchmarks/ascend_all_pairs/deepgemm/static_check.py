"""Stdlib-only structural checks. No tensor/factory/native execution occurs."""

import ast
from collections import Counter
import datetime
import importlib
import importlib.abc
import json
from math import prod
from pathlib import Path
import shutil
import sys

if __package__:
    from .layouts import convolution_plan, group_ranges, moe_ranges
    from .runtime import WORKSPACE, sha256
else:
    from layouts import convolution_plan, group_ranges, moe_ranges
    from runtime import WORKSPACE, sha256

HERE = Path(__file__).resolve().parent
EVIDENCE = WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/deepgemm"
FORBIDDEN = ("torch", "torch_npu", "deep_gemm", "tilelang", "tileops", "acl", "pyacl")


class NoDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise AssertionError(f"Forbidden import during CPU static audit: {fullname}")
        return None


def _contiguous_strides(shape):
    return tuple(prod(shape[i + 1:]) for i in range(len(shape)))


def check_patch_bounds(params):
    plan = convolution_plan(params)
    shape = params["input"][0]
    padded = (shape[0], shape[1], *(s + lo + hi for s, (lo, hi) in zip(shape[2:], plan["pads"])))
    strides = _contiguous_strides(padded)
    # Prove monotone, nonnegative strided view bounds for every group, using
    # the largest reachable batch/channel/output/kernel indices.
    maximum = ((shape[0] - 1) * strides[0] + (shape[1] - 1) * strides[1]
               + sum(((out - 1) * step + (kernel - 1) * dilation) * stride
                     for out, step, kernel, dilation, stride in zip(
                         plan["spatial"], plan["stride"], plan["kernel"], plan["dilation"], strides[2:])))
    assert 0 <= maximum < prod(padded)
    assert prod(plan["output"]) == plan["rows"] * params["weight"][0][0]
    assert plan["reduction"] == prod(params["weight"][0][1:])
    assert tuple(params["output"][0]) == plan["output"]
    return {"output_shape": plan["output"], "max_patch_storage_index": maximum,
            "padded_elements": prod(padded), "packed_group_elements": plan["rows"] * plan["reduction"]}


def main():
    guard = NoDeviceImports()
    sys.meta_path.insert(0, guard)
    checks = []
    files = sorted(HERE.glob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        compile(tree, str(path), "exec")  # compile to in-memory Python bytecode only
    checks.append({"name": "python_ast_and_syntax", "status": "pass", "files": len(files)})
    sys.path.insert(0, str(WORKSPACE / "TileOPs"))
    package = importlib.import_module("benchmarks.ascend_all_pairs.deepgemm")
    sys.path.pop(0)
    assert all(callable(factory) for factory in package.PROVIDERS.values())
    assert not any(name.split(".")[0] in FORBIDDEN for name in sys.modules)
    checks.append({"name": "lazy_provider_import_with_forbidden_import_guard", "status": "pass", "providers": sorted(package.PROVIDERS)})
    coverage = json.loads((HERE / "coverage.json").read_text())
    inventory_path = WORKSPACE / coverage["inventory"]["file"]
    assert sha256(inventory_path) == coverage["inventory"]["sha256"], "Shared inventory changed; regenerate coverage"
    inventory = json.loads(inventory_path.read_text())
    expected = {(c["op"], c["case_id"]) for c in inventory["cases"]}
    observed = {(c["op"], c["case_id"]) for c in coverage["cases"]}
    assert expected == observed and len(observed) == len(coverage["cases"])
    assert {o["op"] for o in inventory["operators"]} == {o["op"] for o in coverage["operators"]}
    assert set(package.PROVIDERS) == {o["op"] for o in coverage["operators"] if o["registered_provider"]}
    for case in coverage["cases"]:
        assert case["numerical_receipt"] is None and case["input_protection_receipt"] is None
        assert case["native_launch_attribution_receipt"] is None
        assert case["status"] not in ("verified", "numerical_failure")
        assert bool(case["adapter"]) == (case["semantic_status"] == "supported_unverified")
    checks.append({"name": "complete_inventory_and_no_fabricated_device_receipts", "status": "pass", "operators": len(coverage["operators"]), "cases": len(observed)})
    evidence = {}
    for operator in coverage["operators"]:
        for ref in operator["source_evidence"] + ([operator["adapter"]] if operator["adapter"] else []):
            evidence[(ref["file"], ref["line"])] = ref
    for (name, number), ref in evidence.items():
        path = WORKSPACE / name
        assert sha256(path) == ref["sha256"], name
        assert 1 <= number <= len(path.read_text().splitlines()), name
    audit = json.loads((HERE / "source_audit.json").read_text())
    for name, digest in audit["source_sha256"].items():
        assert sha256(WORKSPACE / name) == digest, name
    checks.append({"name": "source_line_and_hash_integrity", "status": "pass", "references": len(evidence), "vendor_files": len(audit["source_sha256"])})
    snapshot_path = Path("/opt/tileops-all-pairs/control/source_manifest.json")
    snapshot = json.loads(snapshot_path.read_text())
    original_hashes = {item["path"]: item["sha256"] for item in snapshot["files"]}
    for name, digest in audit["source_sha256"].items():
        assert original_hashes[name] == digest, f"Vendor source changed from initial snapshot: {name}"
    checks.append({"name": "vendor_sources_unchanged_from_supervisor_snapshot", "status": "pass",
                   "files": len(audit["source_sha256"]), "snapshot": str(snapshot_path), "snapshot_sha256": sha256(snapshot_path)})
    adapters = ast.parse((HERE / "adapters.py").read_text())
    banned_math = {"mm", "matmul", "bmm", "einsum", "conv1d", "conv2d", "conv3d", "sum", "mean", "softmax", "silu", "relu", "sigmoid", "mul", "add", "cumsum"}
    for node in ast.walk(adapters):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in banned_math, (node.lineno, node.func.attr)
        assert not isinstance(node, ast.MatMult)
    checks.append({"name": "adapter_has_no_torch_arithmetic_fallback", "status": "pass", "note": "AST exclusion check plus source review; not a numerical or native-launch test."})
    geometry = []
    for case in inventory["cases"]:
        if case["op"] in ("Conv1dFwdOp", "Conv2dFwdOp", "Conv3dFwdOp"):
            geometry.append({"op": case["op"], "case_id": case["case_id"], **check_patch_bounds(case["params"])})
    checks.append({"name": "all_manifest_convolution_geometry_and_view_bounds", "status": "pass", "cases": len(geometry)})
    # Small integer-only boundary cases, covering empty and ragged groups.
    assert group_ranges([0, 1, 0, 3], [0, 0, 1, 1], 4) == [(0, 0), (0, 1), (1, 1), (1, 4)]
    assert moe_ranges({"masked": {"max_m": 4}}, [0, 3], (2, 4, 8), 2) == [(0, 0), (4, 7)]
    assert moe_ranges({"contiguous": {"metadata_kind": "physical_psum", "alignment": 1}}, [0, 3, 4], (4, 8), 3) == [(0, 0), (0, 3), (3, 4)]
    rejected = 0
    for sizes, offsets, total in (([1, 2], [0, 0], 3), ([-1], [0], 1), ([2], [0], 3)):
        try:
            group_ranges(sizes, offsets, total)
        except ValueError:
            rejected += 1
    assert rejected == 3
    checks.append({"name": "integer_metadata_bounds", "status": "pass", "valid_examples": 3, "invalid_examples_rejected": rejected})
    # The finite E4M3 exponent range [-9,8], with <=3 fractional bits, is a
    # strict subset of BF16's range and seven fractional bits. No tensor cast is
    # executed here. Exact widening is a representational argument, not a
    # proof that reassociated scaled GEMM passes the original numeric tolerance.
    checks.append({"name": "e4m3_to_bf16_representation_review", "status": "pass",
                   "finite_e4m3_min_exponent": -9, "finite_e4m3_max_exponent": 8,
                   "source_fraction_bits": 3, "target_fraction_bits": 7,
                   "numeric_verification": "not run"})
    checks.append({"name": "host_cxx_deployment_probe", "status": "blocked",
                   "receipt": str((HERE / "deployment_probe.json").relative_to(WORKSPACE)),
                   "reason": "GCC 11 C++20 <format> compile-only probe fails; no native extension or device code was built/imported."})
    result = {"schema": "deepgemm-static-check/v1", "status": "passed_with_deployment_blocker",
              "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "checks": checks, "convolution_geometry": geometry,
              "device_runs": 0, "factory_runs": 0, "native_imports": 0, "tensor_allocations": 0,
              "not_validated": ["numerical accuracy", "input preservation at runtime", "native launch attribution", "latency", "CANN9.1 kernel compatibility"],
              "source_files": {str(p.relative_to(WORKSPACE)): sha256(p) for p in files}}
    (HERE / "static_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    progress = json.loads((HERE / "progress.json").read_text())
    progress.update(phase="cpu_only_complete", updated_at=result["created_at"],
                    static_checks=str((HERE / "static_checks.json").relative_to(WORKSPACE)),
                    coverage=str((HERE / "coverage.json").relative_to(WORKSPACE)),
                    readme=str((HERE / "README.md").relative_to(WORKSPACE)),
                    remaining=["Compatible C++20 compiler/local native build; CANN9.1 vs source CANN9.20 compatibility unresolved",
                               "Future explicitly authorized NPU full manifest/oracle/input-preservation/native-ownership validation",
                               "Shared factories missing for the nine operators listed in coverage; no tests invented"])
    (HERE / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    for name in ("static_checks.json", "progress.json"):
        shutil.copy2(HERE / name, EVIDENCE / name)
    print(json.dumps({"status": result["status"], "checks": Counter(x["status"] for x in checks),
                      "device_runs": 0, "receipt": str(HERE / "static_checks.json")}, indent=2))


if __name__ == "__main__":
    main()
