"""Rejected diagnostic ACLNN route; providers use tensorlist.py instead.

smoke002 proved that this owned wrapper selected system built-in Foreach binaries.
Its numerical receipts are retained for diagnosis and are not ownership acceptance.
"""
import ctypes as C
import re
from pathlib import Path
from .graph import GraphCall


class HybridCall(GraphCall):
    _native_foreach=True

    def __init__(self,library):
        super().__init__(library)
        self.native_pending=[]
        self.native_functions={}
        self.nnbase=None

    def _native_init(self):
        if self.nnbase is not None:return
        self.nnbase=C.CDLL('libnnopbase.so')
        def bind(name,args,result=C.c_int):
            f=getattr(self.nnbase,name);f.argtypes=args;f.restype=result;return f
        self.native_tensor=bind('aclCreateTensor',[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),
            C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p],C.c_void_p)
        self.native_list=bind('aclCreateTensorList',[C.POINTER(C.c_void_p),C.c_uint64],C.c_void_p)
        self.native_scalar=bind('aclCreateScalar',[C.c_void_p,C.c_int],C.c_void_p)
        self.native_destroy_tensor=bind('aclDestroyTensor',[C.c_void_p])
        self.native_destroy_list=bind('aclDestroyTensorList',[C.c_void_p])
        self.native_destroy_scalar=bind('aclDestroyScalar',[C.c_void_p])

    def _signature(self,op):
        if op in self.native_functions:return self.native_functions[op]
        from benchmarks.ascend_baselines import _owned_native
        api='aclnn'+op
        snake=re.sub(r'(?<!^)(?=[A-Z])','_',op).lower()
        header=self.vendor/'op_api/include'/('aclnn_'+snake+'.h')
        match=re.search(r'\b'+api+r'GetWorkspaceSize\s*\((.*?)\)\s*;',header.read_text(),re.S)
        if not match:raise RuntimeError('No original prototype: '+str(header))
        parameters=match.group(1).split(',')[:-2]
        kinds=[]
        for parameter in parameters:
            kind=next((name for name in ('aclTensorList','aclTensor','aclScalar') if name in parameter),None)
            if kind is None:raise RuntimeError('Unimplemented original ABI parameter: '+parameter)
            kinds.append(kind)
        lib=_owned_native(self.vendor/'op_api/lib/libcust_opapi.so',[api,api+'GetWorkspaceSize'])
        size=getattr(lib,api+'GetWorkspaceSize');size.argtypes=[C.c_void_p]*len(kinds)+[C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];size.restype=C.c_int
        launch=getattr(lib,api);launch.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];launch.restype=C.c_int
        result=(lib,size,launch,kinds);self.native_functions[op]=result;return result

    def _tensor(self,tensor,record):
        if not tensor.is_contiguous():raise ValueError('Native TensorList bridge requires packed tensors.')
        shape=(C.c_int64*tensor.ndim)(*tensor.shape);stride=(C.c_int64*tensor.ndim)(*tensor.stride())
        record['keep'].extend((tensor,shape,stride))
        ptr=self.native_tensor(shape,tensor.ndim,self.codes[tensor.dtype],stride,0,2,shape,tensor.ndim,tensor.data_ptr())
        if not ptr:raise RuntimeError('aclCreateTensor returned null')
        return ptr

    def call(self,op,inputs,outputs,attrs,*,formats=None,names=None,constants=None):
        if not op.startswith('Foreach'):
            return super().call(op,inputs,outputs,attrs,formats=formats,names=names,constants=constants)
        if op not in self.owned_ops:raise RuntimeError('Own vendor lacks '+op+'; refusing fallback')
        if attrs:raise ValueError('Foreach ACLNN interface has no attrs in this adapter')
        self._native_init();lib,size,launch,kinds=self._signature(op)
        values=[*inputs,*outputs]
        if len(values)!=len(kinds):raise ValueError('Original ACLNN prototype/argument count mismatch')
        record=dict(tensors=[],lists=[],scalars=[],keep=[]);self.native_pending.append(record)
        arguments=[];constants=constants or {}
        for i,(value,kind) in enumerate(zip(values,kinds,strict=True)):
            if kind=='aclScalar':
                if i not in constants:raise ValueError('Scalar must have an explicit host constant; no input-derived cache')
                storage=constants[i];record['keep'].append(storage)
                ptr=self.native_scalar(C.cast(storage,C.c_void_p),self.codes[value.dtype])
                if not ptr:raise RuntimeError('aclCreateScalar returned null')
                record['scalars'].append(ptr)
            else:
                tensor=self._tensor(value,record)
                if kind=='aclTensorList':
                    pointers=(C.c_void_p*1)(tensor);record['keep'].append(pointers)
                    ptr=self.native_list(pointers,1)
                    if not ptr:
                        self.native_destroy_tensor(tensor);raise RuntimeError('aclCreateTensorList returned null')
                    # The original C++ example destroys the list, which owns these descriptors.
                    record['lists'].append(ptr)
                else:ptr=tensor;record['tensors'].append(ptr)
            arguments.append(ptr)
        workspace_size=C.c_uint64();executor=C.c_void_p()
        self.check(size(*arguments,C.byref(workspace_size),C.byref(executor)),'aclnn'+op+'GetWorkspaceSize')
        workspace=self.torch.empty(workspace_size.value,device=inputs[0].device,dtype=self.torch.uint8)
        record['keep'].extend((workspace,executor,lib))
        self.check(launch(workspace.data_ptr() if workspace_size.value else None,workspace_size.value,executor,
                          self.torch.npu.current_stream().npu_stream),'aclnn'+op)

    def cleanup(self):
        self.torch.npu.synchronize()
        while self.native_pending:
            record=self.native_pending.pop()
            for p in record['lists']:self.check(self.native_destroy_list(p),'aclDestroyTensorList')
            for p in record['tensors']:self.check(self.native_destroy_tensor(p),'aclDestroyTensor')
            for p in record['scalars']:self.check(self.native_destroy_scalar(p),'aclDestroyScalar')
        super().cleanup()
