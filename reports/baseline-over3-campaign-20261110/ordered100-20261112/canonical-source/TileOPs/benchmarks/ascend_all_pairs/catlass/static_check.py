"""Device-free static checks for the CATLASS adapters and coverage evidence.

Only syntax/import safety, metadata algebra and ELF inspection are checked.
No tensor module, factory, compiled library or device API is executed.
"""
from __future__ import annotations

import ast
import builtins
from collections import Counter
from datetime import datetime, timezone
import importlib
import inspect
import json
from math import prod
from pathlib import Path
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
import audit

HERE, WORKSPACE = audit.HERE, audit.WORKSPACE


def must(condition, detail):
    if not condition:
        raise AssertionError(detail)


def import_safe():
    forbidden = {"torch", "torch_npu", "acl", "aclnn", "catlass", "tilelang", "triton", "cupy", "benchmarks"}
    original = builtins.__import__
    observed = []
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            raise AssertionError(f"Device/parent import forbidden in static inspection: {name}")
        observed.append(name)
        return original(name, *args, **kwargs)
    builtins.__import__ = guarded
    try:
        package = audit.load_package()
        modules = {}
        for name in ("build", "policy", "gemm", "convolution", "attention", "fp8", "int8", "vector", "reduction", "pooling", "mla", "legacy_build"):
            modules[name] = importlib.import_module(package.__name__ + "." + name)
    finally:
        builtins.__import__ = original
    must(not any(x.split(".")[0] in forbidden for x in sys.modules), "Forbidden module already loaded")
    for op, factory in (package.PROVIDERS | package.DEFERRED_PROVIDERS).items():
        must(callable(factory) and list(inspect.signature(factory).parameters) == ["case"], f"Provider signature: {op}")
    must(not package._bindings.exists(), "Current missing-binding assumption changed; regenerate coverage")
    return package, modules, {"passed": True, "imported_adapter_modules": sorted(modules),
                              "forbidden_import_attempts": 0, "factories_executed": 0,
                              "providers": sorted(package.PROVIDERS), "deferred_providers": sorted(package.DEFERRED_PROVIDERS)}


def check_metadata(case, modules):
    p, op = case["params"], case["op"]
    shape = lambda name: tuple(p[name][0])
    dtype = lambda name: p[name][1]
    expected = shape("d" if "d" in p else "o" if "o" in p else "o_slc" if "o_slc" in p else "output")
    route = case["route"]
    if route == "convolution":
        x, w = shape("input"), shape("weight")
        _, _, _, _, out = modules["convolution"].geometry(x[2:], w[2:], p["stride"], p["padding"], p["dilation"])
        g = p["groups"]
        must(x[1] % g == w[0] % g == 0 and x[1] // g == w[1], "Convolution group/channel shape")
        must(expected == (x[0], w[0], *out), "Convolution full output shape")
        must(dtype("input") == dtype("weight"), "Convolution operand dtype")
        if "bias" in p:
            must(shape("bias") == (w[0],) and dtype("bias") == dtype("input"), "Native bias vector contract")
    elif route == "gemm":
        a, b = shape("a"), shape("b")
        must(dtype("a") == dtype("b"), "GEMM input dtype")
        if op == "GemmFwdOp":
            a = a[::-1] if p["trans_a"] else a
            b = b[::-1] if p["trans_b"] else b
            must(a[1] == b[0] and expected == (a[0], b[1]), "GEMM transpose/output shape")
        elif op == "BmmFwdOp":
            must(a[0] == b[0] and a[2] == b[1] and expected == (a[0], a[1], b[2]), "BMM output shape")
        elif op == "GroupedGemmFwdOp":
            groups = shape("batch_sizes")[0]
            must(shape("batch_offsets") == (groups,), "Grouped metadata extent")
            if p["transpose_a"]:
                b = b[::-1] if p["transpose_b"] else b
                must(a[0] == b[0] and expected == (groups, a[1], b[1]), "Grouped transposed A output")
            else:
                k, n = b[-1], b[-2] if p["transpose_b"] else b[-1]
                if not p["transpose_b"]:
                    k = b[-2]
                must(b[0] == groups and a[1] == k and expected == (a[0], n), "Grouped row output")
        else:
            must(expected == (*a[:-1], b[1]) and a[-1] == b[-1], "Expert output shape")
            must(shape("layout_metadata") == (b[0],), "Expert metadata shape")
    elif route in ("fp8", "int8"):
        a, b = shape("a"), shape("b")
        if op == "BmmFP8FwdOp":
            k, n = (b[2], b[1]) if p["trans_b"] else (b[1], b[2])
            must(a[0] == b[0] and a[2] == k and expected == (a[0], a[1], n), "FP8 BMM shape")
            must(prod(shape("scale_a")) == prod(shape("scale_b")) == 1, "FP8 BMM scalar scales")
        elif op == "MoEGroupedGemmFP8FwdOp":
            g, m, k = a
            n = b[1]
            must(g == b[0] and k == b[-1] and k % 128 == 0, "FP8 expert matrix shape")
            must(shape("a_scale") == (g, m, k // 128) and shape("b_scale") == (g, (n + 127) // 128, k // 128), "FP8 expert scale grid")
            must(expected == (g, m, n), "FP8 expert output")
        else:
            m, k = a
            n = b[0]
            must(k == b[1] and expected == (m, n), "Scaled GEMM shape")
            if route == "fp8":
                must(k % 128 == 0, "Native per-block K alignment")
                must(prod(shape("scale_a")) == 1 or shape("scale_a") == (m, k // 128), "FP8 A scale grid")
                must(prod(shape("scale_b")) == 1 or shape("scale_b") in ((n, k // 128), ((n + 127) // 128, k // 128)), "FP8 B scale grid")
            else:
                must(prod(shape("scale_a")) in (1, m) and prod(shape("scale_b")) in (1, n), "INT8 token/channel scale grid")
            if "bias" in p:
                must(shape("bias") == (n,) and dtype("bias") in ("float16", "bfloat16", "float32"), "Native FP8 bias shape/dtype")
    elif route == "attention":
        q = shape("q")
        kn, vn = ("k_pages", "v_pages") if op == "GQAPagedFwdOp" else ("k_new", "v_new") if op == "GQAPrefillPagedWithKVCacheFwdOp" else ("k", "v")
        k, v = shape(kn), shape(vn)
        must(k == v and q[-1] == k[-1] and q[-1] <= 128 and q[-2] % k[-2] == 0, "Original attention QKV head/output width")
        must(dtype("q") == dtype(kn) == dtype(vn) and dtype("q") in ("float16", "bfloat16"), "Native attention dtype")
        must(expected == q, "Complete attention output shape")
        if op == "GQAPrefillPagedWithKVCacheFwdOp":
            must(shape("k_pages") == shape("v_pages") and dtype("k_pages") == dtype("q"), "Append cache output shape/dtype")
            must(p["fuse_rope"] is False, "Append RoPE contract")
    elif route == "mla":
        q, qr, k, kr = (shape(n) for n in ("q", "q_pe", "k", "k_pe"))
        must(q[-1] == 512 and qr == (*q[:-1], 64) and k[-2:] == (1, 512) and kr == (*k[:-1], 64), "MLA complete latent/rope shape")
        must(q[0] == k[0] and q[1] in (64, 128) and k[1] % 128 == 0 and expected == q, "MLA native paging/output contract")
    elif route == "vector":
        inputs = [p[n] for n in ("input", "other") if n in p]
        must(inputs and all(v[1] == inputs[0][1] for v in inputs), "Vector original equal dtype contract")
        dims = [tuple(v[0]) for v in inputs]
        maximum = max(map(len, dims))
        padded = [(1,) * (maximum - len(s)) + s for s in dims]
        result = tuple(max(v) for v in zip(*padded))
        must(all(all(d in (1, r) for d, r in zip(s, result)) for s in padded), "Vector broadcast compatibility")
        must(expected == result, "Vector complete output shape")
    elif route == "sum":
        dims = p["dim"]
        x = shape("x")
        dims = tuple(range(len(x))) if dims is None else (dims,) if isinstance(dims, int) else tuple(dims)
        dims = tuple(d % len(x) for d in dims)
        result = tuple(1 if i in dims else s for i, s in enumerate(x)) if p["keepdim"] else tuple(s for i, s in enumerate(x) if i not in dims)
        must(expected == result, "Full sum output shape")
    elif route == "pooling":
        x = shape("input")
        if op == "AdaptiveMaxPool2dFwdOp":
            result = (*x[:-2], *p["output_size"])
            windows = modules["pooling"].adaptive_windows(x[-2:], p["output_size"])
            must(all(0 < prod(hi - lo for lo, hi in w) <= 64 for w in windows), "Adaptive whole-window64 limit")
        else:
            *_, out = modules["pooling"].geometry(x[2:], p["kernel_size"], p.get("stride"), p["padding"], p["dilation"], p["ceil_mode"])
            result = (*x[:2], *out)
        must(expected == result, "Pooling full output shape")
    else:
        raise AssertionError(f"Uninspected route {route}")


def main():
    checks = {}
    coverage = json.loads((HERE / "coverage.json").read_text())
    inventory = json.loads((HERE / "inventory_snapshot.json").read_text())
    # compile() returns a code object only; no execution/bytecode files.
    checked = []
    for path in sorted(HERE.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        compile(tree, str(path), "exec")
        checked.append(audit.rel(path))
    checks["python_syntax"] = {"passed": True, "files": checked}
    package, modules, checks["lazy_import_guard"] = import_safe()
    keys = {(c["op"], c["case_id"]) for c in coverage["cases"]}
    original_keys = {(c["op"], c["case_id"]) for c in inventory["cases"]}
    must(len(keys) == len(coverage["cases"]) and keys == original_keys, "Inventory cases omitted or duplicated")
    must(len(coverage["operators"]) == 205, "Expected205 operators")
    by_key = {(c["op"], c["case_id"]): c for c in inventory["cases"]}
    for case in coverage["cases"]:
        must(case["params"] == by_key[(case["op"], case["case_id"])]["params"], "Case parameters changed")
        must(case["numerical_receipt"] is None and case["device_runs"] == 0, "False numeric receipt")
        compatible = case["semantic_status"] == "supported"
        must(case["case_accepted_by_provider"] == (compatible and case["op"] in package.PROVIDERS), "Registration mismatch")
    checks["scope_and_status"] = {"passed": True, "operators": 205, "cases": len(keys), "params_unchanged": True,
                                  "missing_factory_ops": coverage["missing_workload_factory_ops"], "device_runs": 0}
    count = Counter()
    for case in coverage["cases"]:
        if case["semantic_status"] == "supported":
            try:
                check_metadata(case, modules)
            except Exception as exc:
                raise AssertionError(f"{case['case_key']}: {exc}") from exc
            count[case["route"]] += 1
    checks["static_contract_shapes"] = {"passed": True, "compatible_cases": sum(count.values()), "routes": dict(count),
                                        "scope": "Manifest metadata algebra only; not numerical execution"}
    refs = []
    for case in coverage["cases"]:
        refs += [*case["sources"], case["manifest_source"]]
        if case["adapter"]:
            refs.append(case["adapter"])
    unique_refs = {(r["file"], r["line"], r["sha256"], r["excerpt"]) for r in refs}
    hashes = {}
    for file, line, sha, excerpt in unique_refs:
        path = WORKSPACE / file
        if file not in hashes:
            hashes[file] = audit.digest(path)
        must(hashes[file] == sha, f"Stale source SHA: {file}")
        must(path.read_text().splitlines()[line - 1].strip() == excerpt, f"Stale source line: {file}:{line}")
    for file, sha in inventory["manifest_sha256"].items():
        must(audit.digest(WORKSPACE / file) == sha, f"Manifest changed: {file}")
    catalog = json.loads((HERE / "source_catalog.json").read_text())
    for entry in catalog["files"]:
        must(audit.digest(WORKSPACE / entry["file"]) == entry["sha256"], f"Vendor source changed: {entry['file']}")
    checks["source_provenance"] = {"passed": True, "source_line_refs": len(unique_refs), "catalog_files": len(catalog["files"]),
                                   "manifest_files": len(inventory["manifest_sha256"]), "all_current_hashes_match": True}
    native = {}
    for kind, receipt in coverage["native_builds"].items():
        must(receipt["status"] == "compiled_not_loaded", f"Native build missing: {kind}")
        path = WORKSPACE / receipt["binary"]
        data = path.read_bytes()
        must(data[:4] == b"\x7fELF" and data[5] == 1 and struct.unpack_from("<H", data, 18)[0] == 183,
             f"Not a little-endian AArch64 ELF: {path}")
        symbols = subprocess.run(["readelf", "--wide", "--dyn-syms", str(path)], check=True, capture_output=True, text=True).stdout
        expected = ({"gemm": ["classic_catlass_gemm_bf16"], "attention_cases": ["classic_catlass_fa_cases_run", "classic_catlass_fa_cases_prepare", "classic_catlass_fa_cases_tiling_size"]}
                    .get(kind, ["allpairs_catlass_" + kind] if kind.startswith("fp8_") or kind == "int8" else ["classic_catlass_" + kind]))
        for symbol in expected:
            must(any(line.split()[-1] == symbol and " UND " not in line for line in symbols.splitlines() if line.split()),
                 f"Native host symbol not defined: {kind}:{symbol}")
        build = json.loads((WORKSPACE / receipt["receipt"]).read_text())
        must(build["source_sha256"] == modules["build"].hashes(modules["build"].sources(kind)), f"Stale compile input: {kind}")
        must(audit.digest(path) == build["binary_sha256"] == receipt["binary_sha256"], f"Stale native binary: {kind}")
        must(not build["device_loaded"] and not build["device_run"], "Build receipt falsely marks device execution")
        native[kind] = {"machine": "AArch64", "defined_host_symbols": expected, "sha256": audit.digest(path),
                        "binary_loaded": False, "runtime_ownership": "pending_device_harness"}
    checks["native_elf_and_build"] = {"passed": True, "artifacts": native}
    legacy = coverage["legacy_compatibility_build"]
    legacy_raw = json.loads((WORKSPACE / legacy["receipt"]).read_text())
    must(legacy_raw["source_sha256"] == modules["legacy_build"].hashes(modules["legacy_build"].sources()), "Stale MLA compatibility build")
    must(legacy_raw["returncode"] != 0 and "MmadAtlasA2MLAQK" in legacy["first_error"], "MLA blocker evidence changed")
    checks["legacy_deployment_failure_evidence"] = {"passed": True, "receipt": legacy["receipt"], "first_error": legacy["first_error"],
                                                    "classification": "deployment_blocked_not_numerical_failure"}
    report = {"schema": "catlass-static-checks/v1", "status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
              "device_access": "forbidden", "device_runs": 0, "torch_imports": 0, "vendor_imports": 0,
              "native_libraries_loaded": 0, "coverage_sha256": audit.digest(HERE / "coverage.json"), "checks": checks,
              "not_checked": ["Numerical equivalence", "Runtime full-output manifest validation", "Runtime input protection",
                              "Real-call library ownership", "NPU runtime compatibility", "Timing/performance", "DSL deployment/compilation"]}
    audit.write_json("static_checks.json", report)
    progress = json.loads((HERE / "progress.json").read_text())
    progress.update(phase="cpu_only_complete", updated_at=report["checked_at"], static_checks="passed",
                    remaining=report["not_checked"] + [f"{len(coverage['missing_workload_factory_ops'])} shared workload factories absent in inventory", "Original catlass/_mlir generated DSL bindings absent", "MLA AtlasA2 dispatch unavailable for3510; compile error preserved"])
    progress["changed_files"] = sorted(audit.rel(p) for p in HERE.rglob("*") if p.is_file())
    audit.write_json("progress.json", progress)
    print(json.dumps({"status": "passed", "operators": 205, "cases": len(keys), "compatible_cases": sum(count.values()),
                      "native_artifacts": len(native), "device_runs": 0}, indent=2))


if __name__ == "__main__":
    main()
