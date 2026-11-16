"""CPU static checks only. Never import tensor/device runtimes or execute kernels."""
import ast
import datetime
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
FORBIDDEN = {"torch", "torch_npu", "flash_mla", "acl", "pyacl", "triton", "tilelang"}


class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise AssertionError("CPU check attempted a forbidden runtime import: " + fullname)


def main():
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, BlockRuntime())
    assert not (FORBIDDEN & set(sys.modules))
    parsed = []
    for path in HERE.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        compile(tree, str(path), "exec")  # bytecode in memory only
        parsed.append(path.name)
    spec = importlib.util.spec_from_file_location("_flashmla_static", HERE / "__init__.py", submodule_search_locations=[str(HERE)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = sys.modules["_flashmla_static.contract"]
    adapters = sys.modules["_flashmla_static.adapters"]
    assert set(module.PROVIDERS) == contract.ROUTES
    assert all(callable(value) for value in module.PROVIDERS.values())
    assert not (FORBIDDEN & set(sys.modules))
    sys.path.insert(0, str(WORKSPACE / "TileOPs"))
    from benchmarks.ascend_baselines import UnsupportedCase
    inventory = json.loads((WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/inventory.json").read_text())

    class NativeBoundaryReached(Exception):
        pass

    def stop_before_runtime():
        raise NativeBoundaryReached

    class StaticCase:
        def __init__(self, row):
            self._call = SimpleNamespace(signature=SimpleNamespace(name=row["op"]))
            self.params = row["params"]

        @property
        def inputs(self):
            raise AssertionError("Static factory validation touched case.inputs")

    # Verify actual factories reject unsupported inputs before runtime loading.
    # Accepted cases stop at an explicit boundary; no Candidate/kernel is faked.
    original = adapters.load_native
    adapters.load_native = stop_before_runtime
    accepted, rejected = 0, 0
    for row in inventory["cases"]:
        factory = module.PROVIDERS.get(row["op"])
        if factory is None:
            continue
        supported, _ = contract.classify(row["op"], row["params"])
        try:
            factory(StaticCase(row))
        except NativeBoundaryReached:
            assert supported, (row["op"], row["case_id"])
            accepted += 1
        except UnsupportedCase:
            assert not supported, (row["op"], row["case_id"])
            rejected += 1
        else:
            raise AssertionError("Factory bypassed the native loading boundary")
    adapters.load_native = original
    # Exercise the shape/dtype guard with inert metadata records, never tensors.
    # Its expected slot order comes independently from the original YAML signature.
    import yaml
    specifications = {}
    for operator in inventory["operators"]:
        path = WORKSPACE / "TileOPs" / operator["manifest"]
        if str(path) not in specifications:
            specifications[str(path)] = yaml.safe_load(path.read_text())
    manifest_by_op = {operator["op"]: specifications[str(WORKSPACE / "TileOPs" / operator["manifest"])][operator["op"]]
                      for operator in inventory["operators"]}
    guards_checked = 0
    for row in inventory["cases"]:
        if row["op"] not in module.PROVIDERS or not contract.classify(row["op"], row["params"])[0]:
            continue
        p, signature = row["params"], manifest_by_op[row["op"]]["signature"]
        def metadata(name):
            return SimpleNamespace(shape=tuple(p[name][0]), dtype="torch." + p[name][1])
        args = [metadata(name) if name in p else None for name in signature["inputs"]]
        output_names = [*signature["outputs"], *contract.mutated_inputs(signature, p)]
        results = tuple(metadata(name) for name in output_names)
        result = results[0] if len(results) == 1 else results
        guarded = adapters._manifest_guard(row["op"], p, lambda *args: result)
        assert guarded(*args) is result
        corrupted = [*args]
        corrupted[0] = SimpleNamespace(shape=(*args[0].shape, 1), dtype=args[0].dtype)
        try:
            guarded(*corrupted)
        except ValueError:
            pass
        else:
            raise AssertionError("Manifest input shape mismatch was silently accepted")
        if len(results) > 1:
            guarded = adapters._manifest_guard(row["op"], p, lambda *args: results[0])
            try:
                guarded(*args)
            except ValueError:
                pass
            else:
                raise AssertionError("Partial output was silently accepted")
        guards_checked += 1

    assert contract.axes_plan((4, 128, 4096), (0, 2), False) == ((1,), (0, 2), 16384, (128,))
    assert contract.axes_plan((2, 3, 5), -1, True) == ((0, 1), (2,), 5, (2, 3, 1))
    assert contract.axes_plan((), None) == ((), (), 1, ())
    # Boundary cases exercise the actual integer planner used by the adapter.
    plan = contract.pool_window("MaxPool1dFwdOp", {"kernel_size": 3, "stride": 2, "padding": 1, "dilation": 2}, (7,), (3,), (0,))
    assert plan == ([0, 1, 3], [False, True, True], [False, True, True])
    plan = contract.pool_window("AvgPool1dFwdOp", {"kernel_size": 2, "stride": 2, "padding": 0, "ceil_mode": True, "count_include_pad": True}, (3,), (2,), (1,))
    assert plan == ([2, 2], [True, False], [True, False])
    plan = contract.pool_window("AvgPool1dFwdOp", {"kernel_size": 3, "stride": 2, "padding": 1, "count_include_pad": True}, (3,), (2,), (0,))
    assert plan == ([0, 0, 1], [True, True, True], [False, True, True])
    plan = contract.pool_window("AdaptiveMaxPool2dFwdOp", {}, (5, 5), (2, 2), (1, 0))
    assert plan[0] == [10, 11, 12, 15, 16, 17, 20, 21, 22] and all(plan[1])
    # Scalar arithmetic of the planner is allowed; input values are never reduced.
    visited = 0
    for row in inventory["cases"]:
        if row["op"] not in contract.POOL or not contract.classify(row["op"], row["params"])[0]:
            continue
        p = row["params"]
        spatial, output = p["input"][0][2:], p["output"][0][2:]
        for coord in [tuple(0 for _ in output), tuple(n - 1 for n in output)]:
            ids, included, real = contract.pool_window(row["op"], p, spatial, output, coord)
            assert len(ids) == len(included) == len(real) and any(included)
            bound = 1
            for dim in spatial:
                bound *= dim
            assert all(0 <= i < bound for i in ids)
            visited += 1
    coverage = json.loads((HERE / "coverage.json").read_text())
    assert {(r["op"], r["case_id"]) for r in coverage["cases"]} == {(r["op"], r["case_id"]) for r in inventory["cases"]}
    assert {o["op"] for o in coverage["operators"]} == {o["op"] for o in inventory["operators"]}
    assert all(r["numerical_receipt"] is None and r["status"] != "verified" for r in coverage["cases"])
    for relative, expected in inventory["manifest_sha256"].items():
        import hashlib
        assert hashlib.sha256((WORKSPACE / relative).read_bytes()).hexdigest() == expected, relative
    native = sys.modules["_flashmla_static.native"]
    initial_build = json.loads((native.CACHE / "build_receipt_initial_gcc11_failure.json").read_text())
    assert initial_build["source_sha256"] == native.hashes(native.source_files())
    build = json.loads(native.RECEIPT.read_text())
    edited_files = []
    for relative, expected in build.get("staged_sha256", {}).items():
        staged = WORKSPACE / relative
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == expected
        original = native.VENDOR / str(relative).split("/source/", 1)[1]
        if staged.read_bytes() != original.read_bytes():
            assert original.relative_to(native.VENDOR).as_posix() == "csrc/api/common.h"
            assert staged.read_text() == original.read_text().replace(
                "static constexpr std::string get_dynamic_enum_name(T value)",
                "static inline std::string get_dynamic_enum_name(T value)")
            edited_files.append(str(original.relative_to(native.VENDOR)))
    assert edited_files == ["csrc/api/common.h"]
    assert not (FORBIDDEN & set(sys.modules))
    result = {"schema": "flashmla-cpu-static-check/v1", "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "passed", "nature": "Static/metadata checks only, not tensor numerical validation",
              "syntax_files": parsed, "lazy_provider_count": len(module.PROVIDERS),
              "factory_cases_at_native_boundary": accepted, "factory_cases_rejected_before_runtime": rejected,
              "manifest_metadata_guard_cases": guards_checked,
              "pool_boundary_layouts": visited, "operators": len(coverage["operators"]), "cases": len(coverage["cases"]),
              "device_imports": [], "device_runs": 0, "verified_cases": 0,
              "manifest_hashes_unchanged": True, "full_outputs_and_input_protection": "not_run"}
    result.update(vendor_hashes_unchanged=True, staged_kernel_files_unchanged=True,
                  staged_host_compatibility_only=edited_files)
    payload = json.dumps(result, indent=2) + "\n"
    (HERE / "static_checks.json").write_text(payload)
    evidence = WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/flashmla"
    (evidence / "static_checks.json").write_text(payload)
    progress = json.loads((HERE / "progress.json").read_text())
    progress.update(phase="cpu_scope_complete", updated_at=result["checked_at"],
                    static_checks="static_checks.json", static_check_status="passed",
                    deployment_review="deployment_review.json",
                    runtime_manifest_guards="all declared input/output shapes and dtypes; append caches included",
                    device_runs=0, verified_cases=0)
    progress_text = json.dumps(progress, indent=2) + "\n"
    (HERE / "progress.json").write_text(progress_text)
    (evidence / "progress.json").write_text(progress_text)
    print(payload)


if __name__ == "__main__":
    main()
