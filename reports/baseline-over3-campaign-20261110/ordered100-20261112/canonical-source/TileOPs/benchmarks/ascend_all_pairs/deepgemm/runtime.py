"""Strict, lazy loading of an independently built DeepGEMM extension."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys

WORKSPACE = Path(__file__).resolve().parents[4]
VENDOR = WORKSPACE / "ascend_baselines/DeepGEMM-Ascend"
CACHE = WORKSPACE / ".cache/all-pairs/deepgemm"


class DeploymentBlocked(RuntimeError):
    """A real adapter exists, but the selected local deployment is unavailable."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned_sources():
    files = [VENDOR / "setup.py", VENDOR / "README.md"]
    for directory in ("csrc", "deep_gemm", "third-party/deep-jit/include",
                      "third-party/magic_enum/include"):
        files.extend(p for p in (VENDOR / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".hpp", ".h", ".cpp"))
    return sorted(set(files))


def fingerprint():
    return {str(p.relative_to(WORKSPACE)): sha256(p) for p in owned_sources()}


def load_native(*symbols):
    """Only called by factories, inside the caller's future device reservation."""
    build = Path(os.environ.get("ALLPAIRS_DEEPGEMM_BUILD_ROOT", CACHE / "package")).resolve()
    if not build.is_relative_to(CACHE.resolve()):
        raise DeploymentBlocked("Build root must be inside .cache/all-pairs/deepgemm")
    receipt_path = build / "build_receipt.json"
    if not receipt_path.is_file():
        raise DeploymentBlocked(f"Missing native build receipt: {receipt_path}; see build_host.py")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("machine") != platform.machine() or receipt.get("python_cache_tag") != sys.implementation.cache_tag:
        raise DeploymentBlocked("Native build architecture/Python ABI differs from this process")
    sources = fingerprint()
    if receipt.get("source_sha256") != sources:
        raise DeploymentBlocked("Vendor source differs from the local build receipt")
    binaries = receipt.get("binaries", {})
    if len(binaries) != 1:
        raise DeploymentBlocked("Expected one locally compiled extension in the build receipt")
    for relative, digest in binaries.items():
        binary = (build / relative).resolve()
        if not binary.is_relative_to(build) or not binary.is_file() or sha256(binary) != digest:
            raise DeploymentBlocked("Native extension path/hash failed provenance check")
    package = build / "deep_gemm"
    for src in (VENDOR / "deep_gemm").rglob("*"):
        if src.is_file() and src.suffix in (".py", ".hpp", ".h"):
            target = package / src.relative_to(VENDOR / "deep_gemm")
            if not target.is_file() or sha256(target) != sha256(src):
                raise DeploymentBlocked(f"Build package source mismatch: {target}")
    # Confine future JIT products to this task's cache, before importing the vendor.
    jit = Path(os.environ.get("ALLPAIRS_DEEPGEMM_JIT_CACHE", CACHE / "jit")).resolve()
    if not jit.is_relative_to((WORKSPACE / ".cache/all-pairs").resolve()):
        raise DeploymentBlocked("JIT cache must remain inside the all-pairs cache tree")
    os.environ["DG_JIT_CACHE_DIR"] = str(jit)
    old = sys.modules.get("deep_gemm")
    if old is not None and Path(old.__file__).resolve() != (package / "__init__.py").resolve():
        raise DeploymentBlocked("A different deep_gemm package is already loaded")
    sys.path.insert(0, str(build))
    try:
        module = importlib.import_module("deep_gemm")
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != (package / "__init__.py").resolve():
        raise DeploymentBlocked("deep_gemm resolved outside the selected build")
    if Path(module._C.__file__).resolve() != binary:
        raise DeploymentBlocked("deep_gemm._C differs from the locally compiled extension")
    functions = {}
    for symbol in symbols:
        function = getattr(module._C, symbol, None)
        if not callable(function) or not function.__module__.startswith("deep_gemm._C"):
            raise DeploymentBlocked(f"Missing original C++ binding: {symbol}")
        if hasattr(module, symbol) and getattr(module, symbol) is not function:
            raise DeploymentBlocked(f"Public {symbol} is not the inspected C++ binding")
        functions[symbol] = function
    sources[str(binary.relative_to(WORKSPACE))] = sha256(binary)
    sources[str(receipt_path.relative_to(WORKSPACE))] = sha256(receipt_path)
    return module, functions, sources


def candidate(run, api, adaptation, hashes, *, host_roundtrip=False):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate

    hashes = dict(hashes)
    for name in ("__init__.py", "adapters.py", "layouts.py", "runtime.py", "build_host.py", "reductions.py", "pooling.py"):
        path = Path(__file__).with_name(name)
        hashes[str(path.relative_to(WORKSPACE))] = sha256(path)
    return Candidate(
        Implementation(run), api, adaptation, hashes,
        limitations=("CPU-only review; no device execution, numerical receipt or launch attribution in this audit.",
                     "Requires a build matching this host/CANN and complete manifest/input-preservation checks."),
        timing_mode="host_roundtrip" if host_roundtrip else "device_async",
    )
