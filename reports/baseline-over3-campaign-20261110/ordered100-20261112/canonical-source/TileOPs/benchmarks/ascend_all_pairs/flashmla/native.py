"""Source-pinned deployment loader; importing this file never imports a runtime."""
from hashlib import sha256
import json
from pathlib import Path
import platform

WORKSPACE = Path(__file__).resolve().parents[4]
VENDOR = WORKSPACE / "ascend_baselines/FlashMLA"
CACHE = WORKSPACE / ".cache/all-pairs/flashmla"
RECEIPT = CACHE / "build_receipt.json"


def source_files():
    files = [VENDOR / "setup.py", VENDOR / "csrc/params.h",
             VENDOR / "csrc/cuda_kernels/kv_cache_format.h"]
    for directory in ("csrc/api", "csrc/ascend_kernels", "csrc/3rdparty/kerutils/include", "flash_mla"):
        files.extend(p for p in (VENDOR / directory).rglob("*") if p.suffix in {".py", ".h", ".cpp", ".asc", ".cuh"})
    return sorted(set(files))


def hashes(paths):
    return {str(p.relative_to(WORKSPACE)): sha256(p.read_bytes()).hexdigest() for p in paths}


def deployment_check():
    """Read bytes only. No dlopen, torch import, ACL call or device probe."""
    if not RECEIPT.is_file():
        return False, f"No native build receipt for this host: {RECEIPT}"
    data = json.loads(RECEIPT.read_text())
    if data.get("status") != "compiled":
        return False, data.get("reason", "Native build has not succeeded")
    if data.get("architecture") != platform.machine():
        return False, "Build receipt architecture differs from this host"
    if data.get("source_sha256") != hashes(source_files()):
        return False, "Vendor source changed after the recorded native build"
    for name, digest in data.get("staged_sha256", {}).items():
        staged = WORKSPACE / name
        if not staged.resolve().is_relative_to(CACHE.resolve()) or not staged.is_file() or sha256(staged.read_bytes()).hexdigest() != digest:
            return False, "Staged compatibility source differs from its build receipt"
    binary = WORKSPACE / data["binary"]
    if not binary.resolve().is_relative_to(CACHE.resolve()) or not binary.is_file():
        return False, "Native binary is missing or outside the isolated cache"
    contents = binary.read_bytes()
    if sha256(contents).hexdigest() != data["binary_sha256"]:
        return False, "Native binary hash mismatch"
    expected_machine = {"aarch64": 183, "x86_64": 62}.get(platform.machine())
    if (expected_machine is None or contents[:4] != b"\x7fELF"
            or int.from_bytes(contents[18:20], "little") != expected_machine):
        return False, "Native ELF architecture differs from this host"
    return True, data


def load_native():
    """Runtime-only entry; the caller must separately authorize/reserve its NPU run."""
    ok, result = deployment_check()
    if not ok:
        raise RuntimeError(f"FlashMLA deployment_blocked: {result}")
    import importlib.util
    import torch  # noqa: F401 -- resolves the extension's PyTorch DSOs
    import torch_npu  # noqa: F401 -- runtime-only registration, never used by CPU checks

    binary = WORKSPACE / result["binary"]
    spec = importlib.util.spec_from_file_location("_flashmla_allpairs.npu", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != binary.resolve():
        raise RuntimeError("FlashMLA native module resolved outside its source-pinned build")
    fn = module.sparse_prefill_fwd
    provenance = hashes([*source_files(), binary, RECEIPT, *Path(__file__).parent.glob("*.py")])
    provenance.update(result.get("staged_sha256", {}))
    return fn, provenance
