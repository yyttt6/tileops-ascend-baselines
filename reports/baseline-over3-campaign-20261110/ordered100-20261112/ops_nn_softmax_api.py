"""Complete original Softmax/LogSoftmax C APIs; finite float32 native scope."""
import ctypes as C,hashlib,json,os
from pathlib import Path
class SoftmaxNativeAPI:
 def __init__(self,torch,manifest,qualification=None):
  self.t=torch;self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-softmax-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so')
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p
  self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int
  for op in ['Softmax','LogSoftmax']:
   f=getattr(self.lib,'aclnn'+op+'GetWorkspaceSize');f.argtypes=[C.c_void_p,C.c_int64,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];f.restype=C.c_int
   f=getattr(self.lib,'aclnn'+op);f.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];f.restype=C.c_int
 def __call__(self,x,op,dim):
  t=self.t;assert op in ('Softmax','LogSoftmax') and x.dtype==t.float32 and x.device.type=='npu' and x.is_contiguous();out=t.empty_like(x);keep=[x,out];owned=[]
  def tensor(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);p=self.base.aclCreateTensor(dims,a.ndim,0,strides,0,2,dims,a.ndim,a.data_ptr());assert p;owned.append(p);return p
  try:
   src=tensor(x);dst=tensor(out);size=C.c_uint64();exe=C.c_void_p();rc=getattr(self.lib,'aclnn'+op+'GetWorkspaceSize')(src,int(dim),dst,C.byref(size),C.byref(exe));assert rc==0,(op,'GetWorkspaceSize',rc)
   workspace=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(workspace);rc=getattr(self.lib,'aclnn'+op)(workspace.data_ptr() if workspace is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,(op,'run',rc);t.npu.synchronize();return out
  finally:
   t.npu.synchronize()
   for p in reversed(owned):assert self.base.aclDestroyTensor(p)==0
