"""Original ops-transformer rotary API through its own deployed host library."""

import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _owned_native, source_hashes,
)
from benchmarks.ascend_transform_baselines import _rope_tables


def transformer_rope(case):
    import os
    import torch
    from benchmarks.api import Implementation

    x, = case.inputs
    if x.ndim != 2 or case.params["rope_layout"] != "neox":
        raise UnsupportedCase("The first sample uses the 1D NeoX/half layout.")
    vendor = WORKSPACE / ".tools/baselines/opp/ops-transformer/vendors/baseline_transformer_transformer"
    configured = {Path(p).resolve() for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p}
    if vendor.resolve() not in configured:
        raise RuntimeError("Use baseline-run ops-transformer.")
    binary = vendor / "op_api/lib/libcust_opapi.so"
    api = "aclnnRotaryPositionEmbedding"
    lib = _owned_native(binary, [api, api + "GetWorkspaceSize"])
    base = C.CDLL("libnnopbase.so")  # descriptor allocation only
    base.aclCreateTensor.argtypes = [
        C.POINTER(C.c_int64), C.c_uint64, C.c_int, C.POINTER(C.c_int64),
        C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p,
    ]
    base.aclCreateTensor.restype = C.c_void_p
    base.aclDestroyTensor.argtypes = [C.c_void_p]
    base.aclDestroyTensor.restype = C.c_int
    size_fn, launch = getattr(lib, api + "GetWorkspaceSize"), getattr(lib, api)
    size_fn.argtypes = [C.c_void_p] * 3 + [
        C.c_int64, C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_void_p),
    ]
    size_fn.restype = C.c_int
    launch.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
    launch.restype = C.c_int
    codes = {torch.float16: 1, torch.float32: 0, torch.bfloat16: 27}
    pending = []
    rope_base = case.params.get("base", 10000.)

    def check(rc, name):
        if rc:
            raise RuntimeError(f"{name} returned {rc}")

    def descriptor(t, record):
        sizes = (C.c_int64 * t.ndim)(*t.shape)
        strides = (C.c_int64 * t.ndim)(*t.stride())
        record["keepalive"].extend((t, sizes, strides))
        ptr = base.aclCreateTensor(sizes, t.ndim, codes[t.dtype], strides, 0, 2,
                                   sizes, t.ndim, t.data_ptr())
        if not ptr:
            raise RuntimeError("aclCreateTensor returned null")
        record["descriptors"].append(ptr)
        return ptr

    def run(x):
        cos, sin = _rope_tables(torch, x, rope_base)
        # The original API accepts four dimensions. Equal-shaped tables avoid
        # any ambiguous broadcast mode and mode=0 is the half/NeoX pairing.
        inp = x.contiguous().view(1, x.shape[0], 1, x.shape[1])
        cos = torch.cat((cos, cos), dim=-1).view_as(inp)
        sin = torch.cat((sin, sin), dim=-1).view_as(inp)
        out = torch.empty_like(inp)
        record = {"descriptors": [], "keepalive": []}
        pending.append(record)
        inp_d, cos_d, sin_d, out_d = [descriptor(t, record) for t in (inp, cos, sin, out)]
        size, executor = C.c_uint64(), C.c_void_p()
        check(size_fn(inp_d, cos_d, sin_d, 0, out_d, C.byref(size), C.byref(executor)), api + "GetWorkspaceSize")
        workspace = torch.empty(size.value, device=x.device, dtype=torch.uint8) if size.value else None
        record["keepalive"].extend((workspace, executor))
        check(launch(workspace.data_ptr() if workspace is not None else None,
                     size.value, executor, torch.npu.current_stream().npu_stream), api)
        return out.view_as(x)

    def cleanup():
        torch.npu.synchronize()
        while pending:
            record = pending.pop()
            for ptr in record["descriptors"]:
                check(base.aclDestroyTensor(ptr), "aclDestroyTensor")

    source = LIBRARIES / "ops-transformer/posembedding/rotary_position_embedding"
    deployed = [p for p in vendor.rglob("*") if p.is_file() and (
        p.suffix in (".so", ".json") or "rotary_position_embedding" in p.name)]
    return Candidate(
        Implementation(run), "ops-transformer::" + api,
        "1D -> BSND, mode=half; per-call frequency tables/descriptors/workspace; own deployed API.",
        source_hashes(Path(__file__), Path(__file__).with_name("ascend_transform_baselines.py"),
                      binary, *deployed, *source.glob("op_host/op_api/*"),
                      *source.glob("op_kernel/arch35/*.h")),
        limitations=("Own host API is confirmed; actual device kernel attribution still requires a launch trace.",),
        cleanup=cleanup,
    )
