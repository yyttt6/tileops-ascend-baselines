"""Thin host bindings for original ops-math APIs; no tensor math fallback."""

import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, WORKSPACE, _add_layout, _owned_native, source_hashes,
)


def opsmath_add(case):
    import os
    import torch
    from benchmarks.api import Implementation

    x, _, shape, _, _, pack = _add_layout(case)
    codes = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 27}
    if x.dtype not in codes:
        raise ValueError("This first host bridge supports equal FP32/FP16/BF16 inputs.")
    vendor = WORKSPACE / ".tools/baselines/opp/ops-math/vendors/baseline_math_math"
    configured = {Path(p).resolve() for p in
                  os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p}
    if vendor.resolve() not in configured:
        raise RuntimeError("Select this library's OPP with baseline-run ops-math.")
    binary = vendor / "op_api/lib/libcust_opapi.so"
    lib = _owned_native(binary, ["aclnnAddGetWorkspaceSize", "aclnnAdd"])
    base = C.CDLL("libnnopbase.so")  # descriptor helpers, never the Add implementation
    base.aclCreateTensor.argtypes = [
        C.POINTER(C.c_int64), C.c_uint64, C.c_int, C.POINTER(C.c_int64),
        C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p,
    ]
    base.aclCreateTensor.restype = C.c_void_p
    base.aclDestroyTensor.argtypes = [C.c_void_p]
    base.aclDestroyTensor.restype = C.c_int
    base.aclCreateScalar.argtypes = [C.c_void_p, C.c_int]
    base.aclCreateScalar.restype = C.c_void_p
    base.aclDestroyScalar.argtypes = [C.c_void_p]
    base.aclDestroyScalar.restype = C.c_int
    lib.aclnnAddGetWorkspaceSize.argtypes = [
        C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
        C.POINTER(C.c_uint64), C.POINTER(C.c_void_p),
    ]
    lib.aclnnAddGetWorkspaceSize.restype = C.c_int
    lib.aclnnAdd.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
    lib.aclnnAdd.restype = C.c_int
    pending = []

    def check(rc, api):
        if rc:
            raise RuntimeError(f"{api} returned {rc}")

    def descriptor(tensor, record):
        dims = (C.c_int64 * tensor.ndim)(*tensor.shape)
        strides = (C.c_int64 * tensor.ndim)(*tensor.stride())
        # The pointer starts at this view's first element, so descriptor offset
        # is zero. Inputs were made contiguous by pack, including any broadcast.
        record["keepalive"].extend((tensor, dims, strides))
        ptr = base.aclCreateTensor(
            dims, tensor.ndim, codes[tensor.dtype], strides, 0, 2,
            dims, tensor.ndim, tensor.data_ptr(),
        )
        if not ptr:
            raise RuntimeError("aclCreateTensor returned null")
        record["tensors"].append(ptr)
        return ptr

    def run(x, y):
        a, b = pack(x), pack(y)
        out = torch.empty_like(a)
        # Register before acquiring native resources; keep every call alive
        # until cleanup synchronizes, including failed launches and workspaces.
        record = {"tensors": [], "scalar": None, "keepalive": []}
        pending.append(record)
        ad, bd, od = (descriptor(t, record) for t in (a, b, out))
        alpha = C.c_float(1)
        record["keepalive"].append(alpha)
        scalar = base.aclCreateScalar(C.byref(alpha), 0)
        record["scalar"] = scalar
        if not scalar:
            raise RuntimeError("aclCreateScalar returned null")
        size, executor = C.c_uint64(), C.c_void_p()
        check(lib.aclnnAddGetWorkspaceSize(ad, bd, scalar, od, C.byref(size),
                                          C.byref(executor)), "aclnnAddGetWorkspaceSize")
        workspace = torch.empty(size.value, dtype=torch.uint8, device=x.device) if size.value else None
        record["keepalive"].extend((workspace, executor))
        check(lib.aclnnAdd(workspace.data_ptr() if workspace is not None else None,
                          size.value, executor, torch.npu.current_stream().npu_stream), "aclnnAdd")
        return out.view(shape)

    def cleanup():
        # A failed drain must not free resources still used by the device.
        torch.npu.synchronize()
        while pending:
            record = pending.pop()
            for ptr in record["tensors"]:
                check(base.aclDestroyTensor(ptr), "aclDestroyTensor")
            if record["scalar"]:
                check(base.aclDestroyScalar(record["scalar"]), "aclDestroyScalar")

    root = LIBRARIES / "ops-math/math/add"
    kernel_dir = vendor / "op_impl/ai_core/tbe/kernel/ascend950/add_apt"
    dtype_name = {torch.float32: "FLOAT32", torch.float16: "FLOAT16",
                  torch.bfloat16: "BFLOAT16"}[x.dtype]
    deployed = [kernel_dir / f"Add_{dtype_name}_high_performance.o",
                kernel_dir / f"Add_{dtype_name}_high_performance.json"]
    return Candidate(
        Implementation(run), "ops-math::aclnnAddGetWorkspaceSize + aclnnAdd",
        "Original two-stage API; broadcast/contiguous views; per-call descriptors and workspace.",
        source_hashes(
            Path(__file__), binary, *deployed, root / "op_api/aclnn_add.h",
            root / "op_api/aclnn_add.cpp", root / "op_api/add.cpp",
            root / "op_kernel/arch35/add.cpp", root / "op_kernel/arch35/add_dag.h",
        ),
        limitations=("Own host entry is checked; actual selected device kernel still needs runtime attribution.",),
        cleanup=cleanup,
    )
