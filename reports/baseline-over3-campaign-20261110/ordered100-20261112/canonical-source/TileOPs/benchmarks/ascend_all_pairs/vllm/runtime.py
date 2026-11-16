"""Lazy source-bound runtime. Importing this module never imports a device runtime."""
from __future__ import annotations

import ast
import hashlib
import importlib
import os
from pathlib import Path
import sys
import types

WORKSPACE = Path(__file__).resolve().parents[4]
VENDOR = WORKSPACE / "ascend_baselines/vllm-ascend"
CACHE = WORKSPACE / ".cache/all-pairs/vllm"
TRITON_ROOT = "vllm_ascend/ops/triton/"


def hashes(*paths):
    result = {}
    for path in paths:
        path = Path(path)
        result[str(path.relative_to(WORKSPACE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def candidate(run, api, adaptation, sources, *, limitations=(), timing_mode="device_async"):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    from benchmarks.ascend_all_pairs.jit_evidence import triton_run

    source_hashes = hashes(Path(__file__), Path(__file__).with_name("contracts.py"),
                           Path(__file__).with_name("__init__.py"), *sources)

    return Candidate(
        Implementation(triton_run(run, source_hashes, CACHE)), "vllm-ascend::" + api, adaptation,
        source_hashes,
        limitations=("CPU static review only; complete manifest outputs, input protection and native launch attribution remain unverified on ARM64/CANN9.1.", *limitations),
        timing_mode=timing_mode,
    )


def reject(reason):
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase(reason)


def validate(case):
    from .contracts import decide
    status, reason = decide(case._call.signature.name, case.params)
    if status not in ("supported_unverified", "deployment_blocked"):
        reject(reason)


def bindings():
    # Called only by a factory under a future, externally managed reservation.
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("deployment_blocked: original Triton Ascend runtime/backend is not available") from exc
    return {"triton": triton, "tl": tl, "HAS_TRITON": True}


def rewrite_imports(tree, injected, module_name):
    """Rebind framework imports only. All bodies/decorators/schedules stay intact."""
    allowed = {"vllm.triton_utils", "vllm.utils.math_utils", "vllm_ascend",
               "vllm_ascend.ops.triton.triton_utils"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in allowed:
            for name in node.names:
                if name.name not in injected:
                    raise RuntimeError(f"Unresolved original import {node.module}.{name.name}")
            node.module = module_name
    return tree


def module(relative, *, hardware=False):
    b = bindings()
    sources = [VENDOR / relative]
    if hardware:
        hw, hw_sources = hardware_module(b)
        b.update({name: getattr(hw, name) for name in
                  ("get_ub_size_bytes", "get_vectorcore_num", "get_aicore_num",
                   "extract_slice", "insert_slice", "init_device_properties_triton")})
        sources += hw_sources
    loaded, generated = _load_one(relative, b)
    return loaded, [*sources, generated]


def _load_one(relative, b):
    source = VENDOR / relative
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    name = "_allpairs_vllm_" + relative.replace("/", "_").replace(".", "_") + key
    bind_name = name + "_bindings"
    injected = types.ModuleType(bind_name)
    injected.__dict__.update(b)
    sys.modules[bind_name] = injected
    tree = rewrite_imports(ast.parse(source.read_text(), str(source)), b, bind_name)
    folder = CACHE / "runtime" / name
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / source.name
    target.write_text(ast.unparse(tree) + "\n")
    loaded = types.ModuleType(name)
    loaded.__file__ = str(target)
    sys.modules[name] = loaded
    exec(compile(target.read_text(), str(target), "exec"), loaded.__dict__)
    return loaded, target


def hardware_module(b):
    envs, env_target = _load_one("vllm_ascend/envs.py", {})
    hw, hw_target = _load_one(TRITON_ROOT + "triton_utils.py", b | {"envs": envs})
    hw.init_device_properties_triton()
    return hw, [VENDOR / "vllm_ascend/envs.py", VENDOR / (TRITON_ROOT + "triton_utils.py"), env_target, hw_target]


def kda_module():
    b = bindings()
    hw, sources = hardware_module(b)
    b.update(cdiv=b["triton"].cdiv, next_power_of_2=b["triton"].next_power_of_2,
             get_vectorcore_num=hw.get_vectorcore_num)
    originals = sorted((VENDOR / (TRITON_ROOT + "kda")).glob("*.py"))
    key = hashlib.sha256(b"".join(p.read_bytes() for p in originals)).hexdigest()[:20]
    name = "_allpairs_vllm_kda_" + key
    bind_name = name + "_bindings"
    injected = types.ModuleType(bind_name)
    injected.__dict__.update(b)
    sys.modules[bind_name] = injected
    folder = CACHE / "runtime" / name
    folder.mkdir(parents=True, exist_ok=True)
    generated = []
    for source in originals:
        if source.name == "__init__.py":
            continue
        tree = rewrite_imports(ast.parse(source.read_text(), str(source)), b, bind_name)
        target = folder / source.name
        target.write_text(ast.unparse(tree) + "\n")
        generated.append(target)
    package = types.ModuleType(name)
    package.__path__ = [str(folder)]
    sys.modules[name] = package
    return importlib.import_module(name + ".kda"), [*sources, *originals, *generated]


def kernel_module(relative, names):
    """Bind intact original JIT entry definitions without distributed registration.

    Selection changes module packaging only; the checker compares the complete
    selected ASTs (including decorators and launch-time constexpr signatures).
    """
    b = bindings()
    hw, sources = hardware_module(b)
    source = VENDOR / relative
    tree = ast.parse(source.read_text(), str(source))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise RuntimeError("Original JIT entry is missing")
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    name = "_allpairs_vllm_selected_" + key
    target = CACHE / "runtime" / name / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ast.unparse(ast.Module(body=selected, type_ignores=[])) + "\n")
    loaded = types.ModuleType(name)
    loaded.__file__ = str(target)
    loaded.__dict__.update(b)
    sys.modules[name] = loaded
    exec(compile(target.read_text(), str(target), "exec"), loaded.__dict__)
    return loaded, hw, [source, target, *sources]
