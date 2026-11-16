"""Run the original ReduceMax graph operator through CANN's documented JIT API."""
import ctypes as C
import os
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes


def opsmath_graph_amax(case):
    import torch
    from benchmarks.api import Implementation

    if case.inputs[0].ndim != 2 or case.params["dim"] != -1:
        raise UnsupportedCase("This graph pilot reduces the final axis of packed rows.")
    vendor = WORKSPACE / ".cache/ascend-classic/native-pilot-20261020/ops-math/install/vendors/classicpilot_math"
    if vendor.resolve() not in {
        Path(p).resolve() for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p
    }:
        raise RuntimeError("Configure the isolated original graph vendor before importing Torch.")
    base = C.CDLL("libascendcl.so")
    compiler = C.CDLL("libacl_op_compiler.so")
    pointer_array = C.POINTER(C.c_void_p)

    def bind(lib, name, arguments, result=C.c_int):
        function = getattr(lib, name)
        function.argtypes = arguments
        function.restype = result
        return function

    create_desc = bind(base, "aclCreateTensorDesc",
                       [C.c_int, C.c_int, C.POINTER(C.c_int64), C.c_int], C.c_void_p)
    destroy_desc = bind(base, "aclDestroyTensorDesc", [C.c_void_p], None)
    set_const = bind(base, "aclSetTensorConst", [C.c_void_p, C.c_void_p, C.c_size_t])
    create_buffer = bind(base, "aclCreateDataBuffer", [C.c_void_p, C.c_size_t], C.c_void_p)
    destroy_buffer = bind(base, "aclDestroyDataBuffer", [C.c_void_p])
    create_attr = bind(base, "aclopCreateAttr", [], C.c_void_p)
    destroy_attr = bind(base, "aclopDestroyAttr", [C.c_void_p], None)
    set_bool = bind(base, "aclopSetAttrBool", [C.c_void_p, C.c_char_p, C.c_uint8])
    set_option = bind(compiler, "aclSetCompileopt", [C.c_int, C.c_char_p])
    execute = bind(compiler, "aclopCompileAndExecute", [
        C.c_char_p, C.c_int, pointer_array, pointer_array,
        C.c_int, pointer_array, pointer_array, C.c_void_p,
        C.c_int, C.c_int, C.c_char_p, C.c_void_p,
    ])

    def check(code, name):
        if code:
            raise RuntimeError(f"{name} returned {code}")

    # ACL_OP_JIT_COMPILE, as declared by the installed acl_op_compiler.h.
    check(set_option(10, b"enable"), "aclSetCompileopt(JIT)")
    pending = []
    codes = {torch.float16: 1, torch.float32: 0, torch.bfloat16: 27, torch.int64: 9}

    def run(x):
        x = x.contiguous()
        axes = torch.tensor([1], device=x.device, dtype=torch.int64)
        out = torch.empty((x.shape[0], 1) if case.params["keepdim"] else (x.shape[0],),
                          device=x.device, dtype=x.dtype)
        record = dict(descriptors=[], buffers=[], keep=[x, axes, out], attr=None)
        pending.append(record)
        for tensor in (x, axes, out):
            shape = (C.c_int64 * tensor.ndim)(*tensor.shape)
            descriptor = create_desc(codes[tensor.dtype], tensor.ndim, shape, 2)
            buffer = create_buffer(tensor.data_ptr(), tensor.numel() * tensor.element_size())
            if descriptor:
                record["descriptors"].append(descriptor)
            if buffer:
                record["buffers"].append(buffer)
            if not descriptor or not buffer:
                raise RuntimeError("Graph tensor descriptor or data buffer allocation failed.")
            record["keep"].append(shape)
        # Supply the original operator's constant axes for graph shape inference.
        constant_axis = (C.c_int64 * 1)(1)
        record["keep"].append(constant_axis)
        check(set_const(record["descriptors"][1], C.cast(constant_axis, C.c_void_p),
                        C.sizeof(constant_axis)), "aclSetTensorConst(axes)")
        attr = create_attr()
        if not attr:
            raise RuntimeError("aclopCreateAttr returned null")
        record["attr"] = attr
        check(set_bool(attr, b"keep_dims", case.params["keepdim"]), "aclopSetAttrBool")
        descriptors, buffers = record["descriptors"], record["buffers"]
        input_descriptors = (C.c_void_p * 2)(*descriptors[:2])
        input_buffers = (C.c_void_p * 2)(*buffers[:2])
        output_descriptors = (C.c_void_p * 1)(descriptors[2])
        output_buffers = (C.c_void_p * 1)(buffers[2])
        record["keep"].extend((input_descriptors, input_buffers, output_descriptors, output_buffers))
        check(execute(b"ReduceMax", 2, input_descriptors, input_buffers, 1,
                      output_descriptors, output_buffers,
                      attr, 0, 0, None, torch.npu.current_stream().npu_stream),
              "aclopCompileAndExecute(ReduceMax)")
        return out

    def cleanup():
        torch.npu.synchronize()
        while pending:
            record = pending.pop()
            for descriptor in record["descriptors"]:
                destroy_desc(descriptor)
            for buffer in record["buffers"]:
                check(destroy_buffer(buffer), "aclDestroyDataBuffer")
            if record["attr"]:
                destroy_attr(record["attr"])

    original = LIBRARIES / "ops-math/math/reduce_max"
    files = [p for root in (original, vendor) for p in root.rglob("*")
             if p.is_file() and p.suffix in (".cpp", ".h", ".py", ".json", ".so", ".o")]
    return Candidate(
        Implementation(run), "ops-math::ReduceMax via CANN graph JIT",
        "Original graph operator, fixed manifest shape/axes and keep_dims; CANN host compiler API, unchanged kernel source.",
        source_hashes(Path(__file__), *files),
        limitations=("Only acceptable after logs prove compilation/loading of this vendor's own source.",
                     "SDK host compiler is explicit; builtin device-kernel fallback is not accepted."),
        cleanup=cleanup,
    )
