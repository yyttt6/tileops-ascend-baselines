"""Load only a source-fingerprinted, locally built DeepSelect extension.

The default build directory is task-local and architecture independent. Merely
importing this module does not import torch, DeepSelect, torch_npu, or ACL.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import struct
import sys

WORKSPACE = Path(__file__).resolve().parents[4]
VENDOR = WORKSPACE / "ascend_baselines/DeepSelect"
CACHE = WORKSPACE / ".cache/all-pairs/deepselect"
SOURCE_FILES = (
    "deep_select/__init__.py", "deep_select/__version__.py", "deep_select/interface.py",
    "csrc/api.cpp", "csrc/structs.h", "csrc/dispatch_utils.h",
    "csrc/ascend_kernels/kernel.asc", "csrc/ascend_kernels/kernel.h",
    "csrc/ascend_kernels/config.h",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vendor_hashes():
    paths = [VENDOR / p for p in SOURCE_FILES]
    paths += sorted((VENDOR / "csrc/3rdparty/kerutils/include").rglob("*.h"))
    return {str(p.relative_to(WORKSPACE)): digest(p) for p in paths}


def load_owned():
    """Runtime-only: caller must hold the real host NPU guard reservation."""
    build = Path(os.environ.get("DEEPSELECT_ALL_PAIRS_BUILD", CACHE / "native")).resolve()
    receipt_path = build / "build_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"DeepSelect deployment_blocked: missing locally compiled build receipt {receipt_path}; no installed-name fallback.")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("source_sha256") != vendor_hashes():
        raise RuntimeError("DeepSelect deployment_blocked: build receipt does not match the current vendor sources.")
    if receipt.get("machine") != platform.machine():
        raise RuntimeError("DeepSelect deployment_blocked: build architecture differs from this host.")
    compatibility = receipt.get("compatibility_sha256", {})
    for path, expected in compatibility.items():
        if digest(WORKSPACE / path) != expected:
            raise RuntimeError(f"DeepSelect deployment_blocked: compatibility source changed: {path}")
    native = (build / receipt["extension"]).resolve()
    if not native.is_relative_to(build) or not native.is_file() or digest(native) != receipt.get("extension_sha256"):
        raise RuntimeError("DeepSelect deployment_blocked: missing, escaping, or stale extension artifact.")
    header = native.read_bytes()[:20]
    if header[:4] != b"\x7fELF" or header[4:6] != bytes((2, 1)):
        raise RuntimeError("DeepSelect deployment_blocked: expected a 64-bit little-endian ELF extension.")
    if struct.unpack_from("<H", header, 18)[0] != {"aarch64": 183, "x86_64": 62}.get(platform.machine()):
        raise RuntimeError("DeepSelect deployment_blocked: ELF machine does not match the host.")
    package = build / "deep_select"
    for name in ("__init__.py", "__version__.py", "interface.py"):
        if digest(package / name) != digest(VENDOR / "deep_select" / name):
            raise RuntimeError(f"DeepSelect copied Python source changed: {name}")
    for name, module in tuple(sys.modules.items()):
        if name == "deep_select" or name.startswith("deep_select."):
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(package):
                raise RuntimeError(f"DeepSelect deployment_blocked: conflicting already-loaded module {name}.")
    import torch_npu  # noqa: F401 -- factory-only; never used by CPU audit/build

    if "deep_select" in sys.modules:
        module = sys.modules["deep_select"]
    else:
        spec = importlib.util.spec_from_file_location("deep_select", package / "__init__.py", submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["deep_select"] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for name in tuple(sys.modules):
                if name == "deep_select" or name.startswith("deep_select."):
                    del sys.modules[name]
            raise
    backend = module.interface._backend
    if Path(backend.__file__).resolve() != native or not backend.__name__.endswith("deep_select_npu"):
        raise RuntimeError("DeepSelect ownership failure: actual backend is not the recorded native NPU extension.")
    if Path(module.topk.__code__.co_filename).resolve() != package / "interface.py":
        raise RuntimeError("DeepSelect ownership failure: topk wrapper is not the inspected original source.")
    return module, native, receipt_path, compatibility
