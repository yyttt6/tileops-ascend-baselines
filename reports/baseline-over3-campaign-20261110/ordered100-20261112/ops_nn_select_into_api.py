"""Full native TensorList API binding. Only instantiate after private-kernel qualification.

No Torch arithmetic implements the measured operator. Every run constructs native
ACL descriptors, asks for workspace, allocates output/workspace and calls the
complete original API. The explicit sync protects descriptor lifetimes.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path

class ForeachPairIntoAPI:
    def __init__(self, torch, library, expected_sha256, device_qualification, op):
        proof = json.loads(Path(device_qualification).read_text())
        if proof.get('private_original_device_kernels_verified') is not True:
            raise RuntimeError('Actual private device kernel attribution required')
        if op not in ('ForeachMaximumList','ForeachMinimumList'):
            raise ValueError('Finite original pair APIs only')
        self.op = op
        self.torch = torch
        library = Path(library).resolve()
        if hashlib.sha256(library.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError('Native API library identity changed')
        self.lib = C.CDLL(str(library), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        self.get = getattr(self.lib,'aclnn'+op+'GetWorkspaceSize')
        self.run = getattr(self.lib,'aclnn'+op)
        self.base = C.CDLL('libnnopbase.so')
        self.base.aclCreateTensor.argtypes = [C.POINTER(C.c_int64), C.c_uint64, C.c_int, C.POINTER(C.c_int64), C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p]
        self.base.aclCreateTensor.restype = C.c_void_p
        self.base.aclCreateTensorList.argtypes = [C.POINTER(C.c_void_p), C.c_uint64]
        self.base.aclCreateTensorList.restype = C.c_void_p
        for name in ['aclDestroyTensor', 'aclDestroyTensorList']:
            fn = getattr(self.base, name); fn.argtypes = [C.c_void_p]; fn.restype = C.c_int
        self.get.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)]
        self.get.restype = C.c_int
        self.run.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
        self.run.restype = C.c_int

    @staticmethod
    def check(rc, operation):
        if rc:
            raise RuntimeError(f'{operation} returned {rc}')

    def __call__(self, xs, ys, outputs):
        torch = self.torch
        if not xs or len(xs) != len(ys):
            raise ValueError('Equal nonempty TensorLists required')
        codes = {torch.float16: 1, torch.bfloat16: 27, torch.float32: 0}
        for x, y in zip(xs, ys):
            if x.dtype != y.dtype or x.dtype not in codes or x.shape != y.shape or x.device != y.device or x.device.type != 'npu' or not x.is_contiguous() or not y.is_contiguous():
                raise ValueError('Original native dtype/shape/device/contiguous contract')
        if len(outputs) != len(xs) or any(o.shape!=x.shape or o.dtype!=x.dtype or o.device!=x.device or not o.is_contiguous() for x,o in zip(xs,outputs)):
            raise ValueError('Provided original output contract changed')
        lists, unowned, keep = [], [], []
        def tensorlist(values):
            handles = []
            for t in values:
                dims = (C.c_int64 * t.ndim)(*t.shape)
                strides = (C.c_int64 * t.ndim)(*t.stride())
                keep.extend([dims, strides])
                ptr = self.base.aclCreateTensor(dims, t.ndim, codes[t.dtype], strides, 0, 2, dims, t.ndim, t.data_ptr())
                if not ptr:
                    raise RuntimeError('aclCreateTensor failed')
                handles.append(ptr); unowned.append(ptr)
            array = (C.c_void_p * len(handles))(*handles); keep.append(array)
            ptr = self.base.aclCreateTensorList(array, len(handles))
            if not ptr:
                raise RuntimeError('aclCreateTensorList failed')
            lists.append(ptr)
            for h in handles:
                unowned.remove(h)  # Original SDK: TensorList owns its descriptors.
            return ptr
        try:
            left, right, out = tensorlist(xs), tensorlist(ys), tensorlist(outputs)
            size, executor = C.c_uint64(), C.c_void_p()
            self.check(self.get(left, right, out, C.byref(size), C.byref(executor)), 'GetWorkspaceSize')
            workspace = torch.empty(size.value, dtype=torch.uint8, device=xs[0].device) if size.value else None
            self.check(self.run(workspace.data_ptr() if workspace is not None else None, size.value, executor, torch.npu.current_stream().npu_stream), self.op)
            torch.npu.synchronize()
            return outputs
        finally:
            torch.npu.synchronize()
            for ptr in lists:
                self.check(self.base.aclDestroyTensorList(ptr), 'DestroyTensorList')
            for ptr in unowned:
                self.check(self.base.aclDestroyTensor(ptr), 'DestroyTensor')
