"""Public CANN graph compiler binding for original source-only OPP packages."""
import ctypes as C
import os
from pathlib import Path

from benchmarks.ascend_baselines import WORKSPACE


class GraphCall:
    """Own descriptors until stream synchronization; never substitute an op."""

    def __init__(self, library, vendor_name="allpairs_transformer"):
        import torch

        suffix = {"ops-math": "math", "ops-nn": "nn", "ops-transformer": "transformer"}[library]
        self.vendor = WORKSPACE / ".cache/all-pairs/ops_transformer/install/vendors" / vendor_name
        if self.vendor.resolve() not in {
            Path(p).resolve() for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p
        }:
            raise RuntimeError("Load the isolated original vendor before importing Torch.")
        self.torch = torch
        self.pending = []
        self.host_calls = {}
        self.last_api = 'aclopCompileAndExecute'
        self.codes = {torch.float16: 1, torch.float32: 0, torch.bfloat16: 27,
                      torch.int64: 9, torch.int32: 3, torch.int8: 2, torch.uint8: 4,
                      torch.bool: 12, torch.float8_e4m3fn: 36, torch.float8_e5m2: 35}
        self.codes['packed_int4'] = 29
        self.codes['uint64'] = 10  # ACL_UINT64; used only for typed empty descriptors.
        self.infer_shim = C.CDLL(str(WORKSPACE / '.cache/all-pairs/ops_transformer/libtransformer_infer_shim.so'), mode=C.RTLD_GLOBAL)
        self.base = C.CDLL("libascendcl.so")
        self.compiler = C.CDLL("libacl_op_compiler.so")
        ptrs = C.POINTER(C.c_void_p)

        def bind(lib, name, arguments, result=C.c_int):
            function = getattr(lib, name)
            function.argtypes, function.restype = arguments, result
            return function

        self.desc = bind(self.base, "aclCreateTensorDesc",
                         [C.c_int, C.c_int, C.POINTER(C.c_int64), C.c_int], C.c_void_p)
        self.destroy_desc = bind(self.base, "aclDestroyTensorDesc", [C.c_void_p], None)
        self.name_desc = bind(self.base, "aclSetTensorDescName", [C.c_void_p, C.c_char_p], None)
        self.dynamic_desc = bind(self.base, "aclSetTensorDynamicInput", [C.c_void_p, C.c_char_p])
        self.constant = bind(self.base, "aclSetTensorConst",
                             [C.c_void_p, C.c_void_p, C.c_size_t])
        self.buffer = bind(self.base, "aclCreateDataBuffer", [C.c_void_p, C.c_size_t], C.c_void_p)
        self.destroy_buffer = bind(self.base, "aclDestroyDataBuffer", [C.c_void_p])
        self.attr = bind(self.base, "aclopCreateAttr", [], C.c_void_p)
        self.destroy_attr = bind(self.base, "aclopDestroyAttr", [C.c_void_p], None)
        self.setters = {
            bool: bind(self.base, "aclopSetAttrBool", [C.c_void_p, C.c_char_p, C.c_uint8]),
            int: bind(self.base, "aclopSetAttrInt", [C.c_void_p, C.c_char_p, C.c_int64]),
            float: bind(self.base, "aclopSetAttrFloat", [C.c_void_p, C.c_char_p, C.c_float]),
            str: bind(self.base, "aclopSetAttrString", [C.c_void_p, C.c_char_p, C.c_char_p]),
            tuple: bind(self.base, "aclopSetAttrListInt",
                        [C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int64)]),
        }
        set_option = bind(self.compiler, "aclSetCompileopt", [C.c_int, C.c_char_p])
        self.execute = bind(self.compiler, "aclopCompileAndExecute", [
            C.c_char_p, C.c_int, ptrs, ptrs, C.c_int, ptrs, ptrs,
            C.c_void_p, C.c_int, C.c_int, C.c_char_p, C.c_void_p])
        self.check(set_option(10, b"enable"), "aclSetCompileopt(JIT)")

    @staticmethod
    def check(code, name):
        if code:
            raise RuntimeError(f"{name} returned {code}")

    def call(self, op, inputs, outputs, attrs, *, formats=None, names=None, constants=None):
        dynamic_names = None
        if op == 'GroupedMatmul':
            from .graph_layout import grouped_matmul
            inputs, outputs, attrs, dynamic_names = grouped_matmul(self.torch, inputs, outputs, attrs)
        if op == 'AttentionUpdate':
            count = attrs['sp']
            if len(inputs) != 2 * count:
                raise ValueError('AttentionUpdate requires sp LSE and sp value tensors.')
            dynamic_names = ['lse'] * count + ['go'] * count
            attrs = dict(attrs, __dynamic_input_lse_cnt=count, __dynamic_input_go_cnt=count)
        if op == 'MhcPost' and inputs[1] is None:
            x = inputs[0]
            n = x.shape[-2]
            identity = self.torch.eye(n, dtype=self.torch.float32, device=x.device)
            hres = identity.expand(*x.shape[:-2], n, n).contiguous()
            inputs = [inputs[0], hres, *inputs[2:]]
        self.last_api = 'aclopCompileAndExecute'
        tensors = [*inputs, *outputs]
        formats = formats or [2] * len(tensors)
        names = names or [None] * len(tensors)
        constants = constants or {}
        record = dict(descriptors=[], buffers=[], keep=list(tensors), attr=None)
        self.pending.append(record)
        for index, (tensor, fmt, name) in enumerate(zip(tensors, formats, names, strict=True)):
            if tensor is None:
                shape = None
                descriptor = self.desc(-1, 0, None, -1)
                buffer = self.buffer(None, 0)
            else:
                if not tensor.is_contiguous():
                    raise ValueError("Graph inputs and outputs must have the declared packed layout.")
                shape = (C.c_int64 * tensor.ndim)(*tensor.shape)
                descriptor = self.desc(self.codes[tensor.dtype], tensor.ndim, shape, fmt)
                buffer = self.buffer(tensor.data_ptr(), getattr(tensor, 'buffer_nbytes',
                                     tensor.numel() * tensor.element_size()))
            if descriptor:
                record["descriptors"].append(descriptor)
            if buffer:
                record["buffers"].append(buffer)
            if not descriptor or not buffer:
                raise RuntimeError("Graph descriptor or buffer allocation failed.")
            record["keep"].append(shape)
            if name:
                encoded = name.encode()
                record["keep"].append(encoded)
                self.name_desc(descriptor, encoded)
            if dynamic_names and index < len(inputs) and dynamic_names[index]:
                encoded = dynamic_names[index].encode()
                record['keep'].append(encoded)
                self.check(self.dynamic_desc(descriptor, encoded), 'aclSetTensorDynamicInput')
            if index in constants:
                array = constants[index]
                record["keep"].append(array)
                self.check(self.constant(descriptor, C.cast(array, C.c_void_p), C.sizeof(array)),
                           "aclSetTensorConst")
        record["attr"] = self.attr()
        if not record["attr"]:
            raise RuntimeError("aclopCreateAttr returned null.")
        for name, value in attrs.items():
            name = name.encode()
            record["keep"].append(name)
            if isinstance(value, (list, tuple)):
                array = (C.c_int64 * len(value))(*value)
                record["keep"].append(array)
                code = self.setters[tuple](record["attr"], name, len(value), array)
            else:
                encoded = value.encode() if isinstance(value, str) else value
                record["keep"].append(encoded)
                code = self.setters[type(value)](record["attr"], name, encoded)
            self.check(code, "aclopSetAttr")
        n = len(inputs)
        descriptors, buffers = record["descriptors"], record["buffers"]
        arrays = [(C.c_void_p * len(values))(*values) for values in
                  (descriptors[:n], buffers[:n], descriptors[n:], buffers[n:])]
        record["keep"].extend(arrays)
        self.check(self.execute(op.encode(), n, arrays[0], arrays[1], len(outputs),
                                arrays[2], arrays[3], record["attr"], 0, 0, None,
                                self.torch.npu.current_stream().npu_stream),
                   "aclopCompileAndExecute(" + op + ")")

    def cleanup(self):
        self.torch.npu.synchronize()
        for binding in self.host_calls.values():
            binding.cleanup()
        while self.pending:
            record = self.pending.pop()
            for descriptor in record["descriptors"]:
                self.destroy_desc(descriptor)
            for buffer in record["buffers"]:
                self.check(self.destroy_buffer(buffer), "aclDestroyDataBuffer")
            if record["attr"]:
                self.destroy_attr(record["attr"])
