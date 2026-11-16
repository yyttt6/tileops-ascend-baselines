"""Record actual JIT calls and their loaded bytes without changing computation.

Only wrappers invoked inside the device checker import compiler packages. These
records describe successful host launches; the checker must synchronize and
verify complete outputs before accepting them as numerical evidence.
"""
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile


WORKSPACE = Path(__file__).resolve().parents[3]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def path_key(path):
    path = Path(path).resolve()
    return str(path.relative_to(WORKSPACE)) if path.is_relative_to(WORKSPACE) else str(path)


class Evidence:
    def __init__(self, source_hashes, cache, engine):
        root = Path(cache) / "launch-evidence"
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=f"{engine}-{os.getpid()}-", dir=root))
        self.hashes = source_hashes
        self.engine = engine
        self.calls = []
        self.sequence = 0
        self.file(Path(__file__))

    def file(self, path):
        path = Path(path).resolve()
        digest = sha(path.read_bytes())
        key = path_key(path)
        if key in self.hashes and self.hashes[key] != digest:
            raise RuntimeError(f"JIT evidence source or artifact changed: {key}")
        self.hashes[key] = digest
        return {"path": key, "sha256": digest}

    def blob(self, data, suffix):
        if isinstance(data, str):
            data = data.encode()
        data = bytes(data)
        if not data:
            raise RuntimeError("JIT evidence requires nonempty generated code/binary")
        path = self.directory / (sha(data) + suffix)
        if not path.exists():
            path.write_bytes(data)
        return self.file(path)

    def flush(self):
        self.sequence += 1
        path = self.directory / f"calls-{self.sequence:04d}.json"
        record = {"schema": "all-pairs-jit-calls/v1", "engine": self.engine,
                  "pid": os.getpid(), "successful_host_launches": len(self.calls),
                  "completion": "Device completion and numerical validity require the enclosing checker receipt.",
                  "calls": self.calls}
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        self.file(path)


def tilelang_run(run, source_hashes, cache):
    """Capture the exact loaded shared object for each JITKernel invocation."""
    evidence = Evidence(source_hashes, cache, "tilelang")

    @wraps(run)
    def wrapped(*args, **kwargs):
        from tilelang.jit.kernel import JITKernel
        original = JITKernel.__call__
        evidence.file(inspect.getsourcefile(JITKernel))
        observed = []

        def traced(kernel, *values, **options):
            adapter = kernel.adapter
            path = Path(adapter.libpath).resolve()
            # ctypes CDLL._name is the path actually loaded by this adapter.
            loaded = getattr(getattr(adapter, "lib", None), "_name", None)
            if loaded is None or Path(loaded).resolve() != path:
                raise RuntimeError("TileLang adapter does not identify the expected loaded library")
            binary = evidence.file(path)
            source = evidence.blob(kernel.get_kernel_source(), ".kernel.txt")
            adapter_source = evidence.file(inspect.getsourcefile(type(adapter)))
            result = original(kernel, *values, **options)
            if evidence.file(path) != binary:
                raise RuntimeError("TileLang binary changed during its launch")
            observed.append({"binary": binary, "generated_source": source,
                             "adapter_source": adapter_source,
                             "target": str(kernel.target),
                             "execution_backend": kernel.execution_backend})
            return result

        JITKernel.__call__ = traced
        try:
            result = run(*args, **kwargs)
            if not observed:
                raise RuntimeError("No original TileLang JITKernel launch observed for this candidate")
            evidence.calls.extend(observed)
            evidence.flush()
            return result
        finally:
            JITKernel.__call__ = original

    return wrapped


def triton_run(run, source_hashes, cache):
    """Capture loaded CompiledKernel bytes returned by original JITFunction.run.

    Native extension routes may issue no Triton calls. Such a record explicitly
    has zero JIT launches and needs its separate native extension evidence.
    """
    evidence = Evidence(source_hashes, cache, "triton")

    @wraps(run)
    def wrapped(*args, **kwargs):
        runtime = sys.modules.get("triton.runtime.jit")
        if runtime is None:
            result = run(*args, **kwargs)
            evidence.flush()
            return result
        JITFunction = runtime.JITFunction
        original = JITFunction.run
        evidence.file(inspect.getsourcefile(JITFunction))
        observed = []

        def traced(function, *values, **options):
            kernel = original(function, *values, **options)
            if options.get("warmup", False):
                return kernel
            if kernel is None or getattr(kernel, "function", None) is None:
                raise RuntimeError("Triton call returned no loaded kernel handle")
            raw = kernel.kernel
            if not isinstance(raw, (bytes, bytearray, memoryview)):
                raise RuntimeError("Triton loaded binary is not available as bytes")
            binary = evidence.blob(raw, ".device.bin")
            definition = evidence.file(inspect.getsourcefile(function.fn))
            compiler = evidence.file(inspect.getsourcefile(type(kernel)))
            observed.append({"binary": binary, "definition": definition,
                             "compiler_source": compiler,
                             "name": kernel.name, "jit_key": kernel.hash,
                             "metadata": dict(kernel.metadata._asdict()),
                             "loaded_function": str(kernel.function)})
            return kernel

        JITFunction.run = traced
        try:
            result = run(*args, **kwargs)
            evidence.calls.extend(observed)
            evidence.flush()
            return result
        finally:
            JITFunction.run = original

    return wrapped
