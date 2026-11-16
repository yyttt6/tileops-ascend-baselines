"""Original ops-math reduction and single-window STFT host APIs."""
import ctypes as C
import os
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _owned_native, source_hashes,
)


class _MathCall:
    def __init__(self, api, arguments, *, library="ops-math", host_library=None):
        import torch

        self.torch = torch
        self.pending = []
        suffix = {"ops-math": "math", "ops-nn": "nn", "ops-transformer": "transformer"}[library]
        self.vendor = Path(os.environ.get(
            f"ASCEND_CLASSIC_{suffix.upper()}_VENDOR",
            str(WORKSPACE / ".cache/ascend-classic/native-pilot-20261020" / library
                / f"install/vendors/classicpilot_{suffix}"),
        )).resolve()
        configured = {Path(p).resolve() for p in
                      os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p}
        if self.vendor not in configured:
            raise RuntimeError(f"Select the isolated {library} vendor before importing Torch.")
        self.binary = (Path(host_library).resolve() if host_library else
                       self.vendor / "op_api/lib/libcust_opapi.so")
        self.lib = _owned_native(self.binary, [api, api + "GetWorkspaceSize"])
        self.api = api
        self.size = getattr(self.lib, api + "GetWorkspaceSize")
        self.size.argtypes = arguments + [C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)]
        self.size.restype = C.c_int
        self.launch = getattr(self.lib, api)
        self.launch.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
        self.launch.restype = C.c_int
        self.base = C.CDLL("libnnopbase.so")
        self.base.aclCreateTensor.argtypes = [
            C.POINTER(C.c_int64), C.c_uint64, C.c_int, C.POINTER(C.c_int64),
            C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p,
        ]
        self.base.aclCreateTensor.restype = C.c_void_p
        self.base.aclDestroyTensor.argtypes = [C.c_void_p]
        self.base.aclDestroyTensor.restype = C.c_int
        self.base.aclCreateIntArray.argtypes = [C.POINTER(C.c_int64), C.c_uint64]
        self.base.aclCreateIntArray.restype = C.c_void_p
        self.base.aclDestroyIntArray.argtypes = [C.c_void_p]
        self.base.aclDestroyIntArray.restype = C.c_int
        self.codes = {torch.float32: 0, torch.float16: 1, torch.int32: 3,
                      torch.int64: 9, torch.complex64: 16, torch.bfloat16: 27,
                      torch.float8_e4m3fn: 36}

    @staticmethod
    def check(rc, api):
        if rc:
            raise RuntimeError(f"{api} returned {rc}")

    def record(self):
        item = {"tensors": [], "arrays": [], "keep": []}
        self.pending.append(item)
        return item

    def tensor(self, tensor, record, *, format_code=2):
        if not tensor.is_contiguous():
            raise ValueError("This host bridge expects packed tensors.")
        shape = (C.c_int64 * tensor.ndim)(*tensor.shape)
        stride = (C.c_int64 * tensor.ndim)(*tensor.stride())
        record["keep"].extend((tensor, shape, stride))
        ptr = self.base.aclCreateTensor(shape, tensor.ndim, self.codes[tensor.dtype],
                                       stride, 0, format_code, shape, tensor.ndim, tensor.data_ptr())
        if not ptr:
            raise RuntimeError("aclCreateTensor returned null")
        record["tensors"].append(ptr)
        return ptr

    def array(self, values, record):
        storage = (C.c_int64 * len(values))(*values)
        record["keep"].append(storage)
        ptr = self.base.aclCreateIntArray(storage, len(values))
        if not ptr:
            raise RuntimeError("aclCreateIntArray returned null")
        record["arrays"].append(ptr)
        return ptr

    def call(self, arguments, record, device):
        size, executor = C.c_uint64(), C.c_void_p()
        self.check(self.size(*arguments, C.byref(size), C.byref(executor)),
                   self.api + "GetWorkspaceSize")
        workspace = self.torch.empty(size.value, device=device, dtype=self.torch.uint8)
        record["keep"].extend((workspace, executor))
        self.check(self.launch(workspace.data_ptr() if size.value else None,
                               size.value, executor,
                               self.torch.npu.current_stream().npu_stream), self.api)

    def cleanup(self):
        self.torch.npu.synchronize()
        while self.pending:
            record = self.pending.pop()
            for p in record["tensors"]:
                self.check(self.base.aclDestroyTensor(p), "aclDestroyTensor")
            for p in record["arrays"]:
                self.check(self.base.aclDestroyIntArray(p), "aclDestroyIntArray")

    def hashes(self, root):
        sources = [p for p in root.rglob("*")
                   if p.suffix in (".cpp", ".h", ".cc", ".py", ".json")]
        deployed = [p for p in self.vendor.rglob("*")
                    if p.is_file() and p.suffix in (".so", ".o", ".json")]
        return source_hashes(Path(__file__), self.binary, *sources, *deployed)


def opsmath_amax(case):
    import torch
    from benchmarks.api import Implementation

    binding = _MathCall("aclnnAmax", [C.c_void_p, C.c_void_p, C.c_bool, C.c_void_p])
    dim, keep = case.params["dim"], case.params["keepdim"]
    dims = (dim,) if isinstance(dim, int) else tuple(dim)

    def run(x):
        packed = x.contiguous()
        reduced = {d % x.ndim for d in dims}
        shape = [1 if i in reduced else d for i, d in enumerate(x.shape)]
        if not keep:
            shape = [d for i, d in enumerate(shape) if i not in reduced]
        out = torch.empty(shape, device=x.device, dtype=x.dtype)
        record = binding.record()
        binding.call([binding.tensor(packed, record), binding.array(dims, record),
                      keep, binding.tensor(out, record)], record, x.device)
        return out

    return Candidate(
        Implementation(run), "ops-math::aclnnAmax",
        "Original reduction API with manifest dimensions and keepDim flag.",
        binding.hashes(LIBRARIES / "ops-math/math/reduce_max"),
        limitations=("Actual device-kernel selection must be attributed to this vendor.",),
        cleanup=binding.cleanup,
    )


def opsmath_fft(case):
    import torch
    from benchmarks.api import Implementation

    x, = case.inputs
    if x.ndim != 1 or x.dtype != torch.complex64:
        raise UnsupportedCase("This sample uses one complete complex64 window.")
    n = x.numel()
    binding = _MathCall("aclStft", [C.c_void_p] * 3 + [C.c_int64] * 3 + [C.c_bool] * 3)

    def run(x):
        out = torch.empty((n, 1), dtype=x.dtype, device=x.device)
        record = binding.record()
        binding.call([binding.tensor(x.contiguous(), record), None,
                      binding.tensor(out, record), n, n, n, False, False, True],
                     record, x.device)
        return out.reshape(x.shape)

    return Candidate(
        Implementation(run), "ops-math::aclStft (single complete complex window)",
        "n_fft=win_length=hop_length=N; rectangular window; full unnormalized complex output; reshape.",
        binding.hashes(LIBRARIES / "ops-math/math/stft"),
        limitations=("Original complex64 STFT dispatches to AI CPU; require own AI CPU binary attribution.",
                     "Keep AI CPU execution separate from AI Core latency comparisons."),
        timing_mode="aicpu", cleanup=binding.cleanup,
    )
