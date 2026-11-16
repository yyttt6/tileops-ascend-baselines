"""Pack the documented device TensorList header for original graph JIT kernels."""
import re
from .graph import GraphCall


class _ListBuffer:
    """Keep tensor metadata for tiling and supply a separate TensorList data buffer."""
    def __init__(self,tensor,header):
        self.tensor=tensor
        self.header=header
    def __getattr__(self,name):return getattr(self.tensor,name)
    def data_ptr(self):return self.header.data_ptr()
    def numel(self):return self.header.numel()
    def element_size(self):return self.header.element_size()


class TensorListGraphCall(GraphCall):
    _graph_tensorlist=True
    def __init__(self,library):
        super().__init__(library)
        self.list_parameters={}

    def _parameters(self,op):
        if op in self.list_parameters:return self.list_parameters[op]
        snake=re.sub(r'(?<!^)(?=[A-Z])','_',op).lower()
        header=self.vendor/'op_api/include'/('aclnn_'+snake+'.h')
        match=re.search(r'\baclnn'+op+r'GetWorkspaceSize\s*\((.*?)\)\s*;',header.read_text(),re.S)
        if not match:raise RuntimeError('Missing original TensorList prototype: '+str(header))
        result=tuple('aclTensorList' in p for p in match[1].split(',')[:-2])
        self.list_parameters[op]=result
        return result

    def call(self,op,inputs,outputs,attrs,*,formats=None,names=None,constants=None):
        if not op.startswith('Foreach'):
            return super().call(op,inputs,outputs,attrs,formats=formats,names=names,constants=constants)
        values=[*inputs,*outputs];parameters=self._parameters(op)
        if len(values)!=len(parameters):raise ValueError('Original TensorList prototype/argument count mismatch')
        packed=[]
        for value,is_list in zip(values,parameters,strict=True):
            if is_list:
                # AscendC ListTensorDescImpl::ListTensorDecode pointer-only representation:
                # byte offset 24, one tensor in upper 32 bits, dimension zero, sentinel,
                # followed by its current device address. Recreated inside every run.
                # GE validates the buffer extent against the logical tensor's byte
                # size even though this dynamic argument contains a list descriptor.
                # Allocate that full extent; only its documented 32-byte header is read.
                words=max(4,(value.numel()*value.element_size()+7)//8)
                header=self.torch.empty((words,),dtype=self.torch.int64,device=value.device)
                header[:4].copy_(self.torch.tensor([24,1<<32,0xffffffff,value.data_ptr()],
                    dtype=self.torch.int64,device=value.device))
                packed.append(_ListBuffer(value,header))
            else:packed.append(value)
        n=len(inputs)
        return super().call(op,packed[:n],packed[n:],attrs,formats=formats,names=names,constants=constants)
