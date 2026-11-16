"""Source-checked, architecture-independent loader. No runtime imports at module scope."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from datetime import timedelta

if __package__:
    from .jit_provenance import JITProvenance, ProvenanceError
else:  # build_host.py also imports runtime as a standalone CPU module.
    from jit_provenance import JITProvenance, ProvenanceError

WORKSPACE = Path(__file__).resolve().parents[4]
VENDOR = WORKSPACE / "ascend_baselines/DeepEP-Ascend"
CACHE = WORKSPACE / ".cache/all-pairs/deepep"

# Keep evidence on disk; do not reuse an unverified persistent JIT object cache.
_JIT_CACHE_ROOT = None
_JIT_PROCESS = None
_JIT_MODULE = None


def _prepare_owned_jit_cache():
    global _JIT_CACHE_ROOT, _JIT_PROCESS
    if _JIT_CACHE_ROOT is None:
        if any(name == "deep_ep" or name.startswith("deep_ep.") for name in sys.modules):
            raise RuntimeError("deep_ep was imported before the adapter owned its JIT cache")
        CACHE.mkdir(parents=True, exist_ok=True)
        _JIT_CACHE_ROOT = Path(tempfile.mkdtemp(prefix="jit-owned-", dir=CACHE)).resolve()
        _JIT_PROCESS = os.getpid()
    elif (_JIT_PROCESS != os.getpid() or _JIT_MODULE is None or
          sys.modules.get("deep_ep") is not _JIT_MODULE):
        raise RuntimeError("JIT cache ownership changed; use a fresh benchmark process")
    os.environ["EP_JIT_CACHE_DIR"] = str(_JIT_CACHE_ROOT)
    return _JIT_CACHE_ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vendor_sources():
    roots = (VENDOR / "deep_ep", VENDOR / "csrc", VENDOR / "third-party/deep_jit/include")
    files = [VENDOR / "setup.py"]
    for root in roots:
        files.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".hpp", ".cpp"))
    return sorted(files)


def source_snapshot():
    return {str(p.relative_to(WORKSPACE)): digest(p) for p in vendor_sources()}


def checked_build():
    """Reject missing, stale, foreign-architecture or untraceable native builds."""
    build = Path(os.environ.get("DEEPEP_ALL_PAIRS_BUILD", CACHE / "host/lib")).resolve()
    receipt = build / "build_receipt.json"
    if not receipt.is_file():
        raise RuntimeError(f"DeepEP deployment_blocked: no source-verified host build receipt at {receipt}; "
                           "see deepep/build_host.py and deployment.json. This factory never imports an installed fallback.")
    record = json.loads(receipt.read_text())
    if record.get("status") != "built" or record.get("architecture") != platform.machine():
        raise RuntimeError("DeepEP deployment_blocked: build receipt has wrong status or architecture.")
    if record.get("python_abi") != sys.implementation.cache_tag:
        raise RuntimeError("DeepEP deployment_blocked: native build Python ABI differs.")
    if record.get("sources") != source_snapshot():
        raise RuntimeError("DeepEP deployment_blocked: original source changed since the host build.")
    for path in (VENDOR / "deep_ep").rglob("*"):
        if path.is_file() and path.suffix in (".py", ".hpp"):
            built = build / "deep_ep" / path.relative_to(VENDOR / "deep_ep")
            if not built.is_file() or digest(built) != digest(path):
                raise RuntimeError(f"DeepEP deployment_blocked: missing/stale staged source {built}")
    binaries = list((build / "deep_ep").glob("_C*.so"))
    if len(binaries) != 1:
        raise RuntimeError(f"DeepEP deployment_blocked: expected one native extension, found {len(binaries)}")
    native = binaries[0]
    if digest(native) != record.get("binary_sha256"):
        raise RuntimeError("DeepEP deployment_blocked: native binary hash differs from build receipt.")
    elf = native.read_bytes()[:20]
    expected_machine = {"aarch64": 183, "x86_64": 62}.get(platform.machine())
    if (elf[:4] != b"\x7fELF" or len(elf) != 20 or elf[5] != 1 or
            int.from_bytes(elf[18:20], "little") != expected_machine):
        raise RuntimeError("DeepEP deployment_blocked: native ELF is not for this host architecture.")
    return build, native


class Runtime:
    def __init__(self, geometry, build, native, provenance):
        global _JIT_MODULE
        self.provenance = provenance
        self.source_hashes = provenance.source_hashes
        import torch
        import torch_npu  # noqa: F401 -- required to register the HCCL backend
        import torch.distributed as dist

        if dist.is_initialized():
            raise RuntimeError("DeepEP adapter requires ownership of an isolated one-rank process group.")
        sys.path.insert(0, str(build))
        try:
            deep_ep = importlib.import_module("deep_ep")
        finally:
            sys.path.pop(0)
        if Path(deep_ep.__file__).resolve() != build / "deep_ep/__init__.py":
            raise RuntimeError("DeepEP package resolved outside its source-verified build.")
        if Path(deep_ep._C.__file__).resolve() != native:
            raise RuntimeError("DeepEP extension resolved outside its source-verified build.")
        checks = ((deep_ep.EPBuffer, "dispatch", "ep.py"), (deep_ep.EPBuffer, "combine", "ep.py"),
                  (deep_ep.EngramBuffer, "write", "engram.py"), (deep_ep.EngramBuffer, "fetch", "engram.py"))
        for cls, name, file in checks:
            if Path(getattr(cls, name).__code__.co_filename).resolve() != build / "deep_ep/buffers" / file:
                raise RuntimeError(f"DeepEP {cls.__name__}.{name} is not the inspected original API.")
        _JIT_MODULE = deep_ep
        self.torch, self.dist, self.deep_ep = torch, dist, deep_ep
        self.buffer = None
        self.owns_group = False
        self.closed = False
        CACHE.mkdir(parents=True, exist_ok=True)
        self.rendezvous = tempfile.TemporaryDirectory(prefix="rank0-", dir=CACHE)
        try:
            dist.init_process_group(
                backend="hccl", init_method=(Path(self.rendezvous.name) / "store").as_uri(),
                rank=0, world_size=1, timeout=timedelta(seconds=60),
            )
            self.owns_group = True
            if geometry.get("kind") == "engram":
                sizes = deep_ep.EngramBuffer.get_storage_size_hint(
                    dist.group.WORLD, [geometry["entries"]], geometry["hidden"], geometry["tokens"], 1)
                self.buffer = deep_ep.EngramBuffer(
                    dist.group.WORLD, num_gpu_bytes=sizes[0], num_rdma_storage_bytes=sizes[1],
                    use_cpu_rdma_storage=False, num_allocated_qps=0, qp_depth=0,
                    explicitly_destroy=True, use_huge1g_page=False, num_gpu_timeout_secs=60)
                self.buffer.set_config([geometry["entries"]], geometry["hidden"], geometry["tokens"], 1)
            else:
                self.buffer = deep_ep.EPBuffer(
                    dist.group.WORLD, num_max_tokens_per_rank=geometry["tokens"], hidden=geometry["hidden"],
                    num_topk=geometry["topk"], deterministic=False, explicitly_destroy=True,
                    num_cpu_timeout_secs=60, num_gpu_timeout_secs=60,
                    use_huge1g_page=False,
                )
        except BaseException as exc:
            self.provenance.poisoned = f"{type(exc).__name__}: {exc}"
            # A failed init may still leave an owned group; no foreign group
            # existed before we began this isolated setup.
            self.owns_group = dist.is_initialized()
            self.close()
            raise

    def observed_run(self, run):
        self.torch.npu.synchronize()
        self.provenance.checkpoint("factory_preparation")
        return self.provenance.wrap(run, self.torch.npu.synchronize)

    def close(self):
        if self.closed:
            return
        failure = None
        try:
            self._close_native()
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.provenance.finish(error=failure)

    def _close_native(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.torch.npu.synchronize()
            if self.buffer is not None:
                self.buffer.destroy()
                self.buffer = None
        finally:
            try:
                self.deep_ep.destroy_all_managed_hccl_comm()
            finally:
                try:
                    if self.owns_group and self.dist.is_initialized():
                        self.dist.destroy_process_group()
                finally:
                    self.rendezvous.cleanup()


def open_runtime(geometry, identity):
    # DeepJIT caches Kernel objects in C++ and exposes no reset/loaded-object API.
    # An already-used process cannot demonstrate a fresh case-to-load relation.
    if _JIT_CACHE_ROOT is not None:
        raise ProvenanceError("DeepEP JIT provenance requires one candidate per fresh process")
    build, native = checked_build()
    root = _prepare_owned_jit_cache()
    files = [*Path(__file__).parent.glob("*.py"), native, build / "build_receipt.json"]
    sources = source_snapshot()
    sources.update({str(p.relative_to(WORKSPACE)) if p.is_relative_to(WORKSPACE) else str(p):
                    digest(p) for p in files})
    provenance = JITProvenance(root, WORKSPACE, VENDOR, build, identity, sources)
    provenance.start()
    runtime = None
    try:
        runtime = Runtime(geometry, build, native, provenance)
        runtime.torch.npu.synchronize()
        provenance.checkpoint("initialization")
        return runtime
    except BaseException as exc:
        provenance.poisoned = f"{type(exc).__name__}: {exc}"
        if runtime is not None:
            runtime.close()
        else:
            provenance.finish(error=provenance.poisoned)
        raise
