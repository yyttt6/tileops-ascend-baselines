"""Complete original aclnnDiv two-stage API; no Torch arithmetic fallback."""
import ctypes as C,json,hashlib,os
from pathlib import Path
class DivAPI:
 def __init__(self,torch):
  self.t=torch;self.N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-div-v1');self.m=json.loads((self.N/'DIV_PACKAGE.json').read_text());self.root=Path(self.m['vendor']);self.pending=[];self.codes={torch.float16:1,torch.float32:0}
  for name,r in self.m['files'].items():assert hashlib.sha256((self.N/'graph-private-v1'/name).read_bytes()).hexdigest()==r['sha256'],name
  assert self.root.resolve() in {Path(p).resolve() for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if p}
  self.lib=C.CDLL(str(self.root/'op_api/lib/libcust_opapi.so'),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so')
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p
  self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int
  self.lib.aclnnDivGetWorkspaceSize.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnDivGetWorkspaceSize.restype=C.c_int
  self.lib.aclnnDiv.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnDiv.restype=C.c_int
 def check(self,rc,name):
  if rc:raise RuntimeError(f'{name} returned {rc}')
 def descriptor(self,t,r):
  dims=(C.c_int64*t.ndim)(*t.shape);strides=(C.c_int64*t.ndim)(*t.stride());r['keep'].extend([t,dims,strides]);p=self.base.aclCreateTensor(dims,t.ndim,self.codes[t.dtype],strides,0,2,dims,t.ndim,t.data_ptr());assert p,'aclCreateTensor null';r['ptrs'].append(p);return p
 def __call__(self,x,y):
  assert x.dtype==y.dtype and x.dtype in self.codes
  a=x.contiguous();b=y.contiguous();out=self.t.empty(self.t.broadcast_shapes(a.shape,b.shape),dtype=x.dtype,device=x.device);r={'keep':[],'ptrs':[]};self.pending.append(r);ad,bd,od=[self.descriptor(t,r) for t in [a,b,out]];size=C.c_uint64();executor=C.c_void_p();self.check(self.lib.aclnnDivGetWorkspaceSize(ad,bd,od,C.byref(size),C.byref(executor)),'aclnnDivGetWorkspaceSize');workspace=self.t.empty(size.value,dtype=self.t.uint8,device=x.device) if size.value else None;r['keep'].extend([workspace,executor]);self.check(self.lib.aclnnDiv(workspace.data_ptr() if workspace is not None else None,size.value,executor,self.t.npu.current_stream().npu_stream),'aclnnDiv');return out
 def cleanup(self):
  self.t.npu.synchronize()
  while self.pending:
   r=self.pending.pop()
   for ptr in r['ptrs']:self.check(self.base.aclDestroyTensor(ptr),'aclDestroyTensor')
