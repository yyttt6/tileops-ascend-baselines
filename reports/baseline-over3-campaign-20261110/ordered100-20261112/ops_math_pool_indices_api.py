"""Complete original aclnnMaxDim returning both values and original int64 indices."""
from pathlib import Path
import ctypes as C,json,hashlib,os
from ops_math_l1_api import MathL1API
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-pool-indices-v1')
class MaxDimAPI(MathL1API):
 def __init__(self,torch):
  self.t=torch;self.pending=[];self.codes={torch.float32:0,torch.float16:1,torch.bfloat16:27,torch.int64:9};mf=N/'MATH_POOL_INDICES_PACKAGE.json';self.m=json.loads(mf.read_text());self.root=Path(self.m['vendor']);assert self.root.resolve() in {Path(p).resolve() for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if p};self.sources={str(mf):hashlib.sha256(mf.read_bytes()).hexdigest()}
  for name,r in self.m['files'].items():
   f=Path(self.m['package'])/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];self.sources[str(f)]=r['sha256']
  self.lib=C.CDLL(str(self.root/'op_api/lib/libcust_opapi.so'),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so');self.get=self.lib.aclnnMaxDimGetWorkspaceSize;self.run=self.lib.aclnnMaxDim;self.get.argtypes=[C.c_void_p,C.c_int64,C.c_bool,C.c_void_p,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.get.restype=C.c_int;self.run.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.run.restype=C.c_int
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int
 def __call__(self,x):
  assert x.ndim==2 and x.dtype in (self.t.float16,self.t.bfloat16,self.t.float32);r={'keep':[],'tensors':[],'arrays':[]};self.pending.append(r);x=x.contiguous();value=self.t.empty((x.shape[0],),dtype=x.dtype,device=x.device);index=self.t.empty((x.shape[0],),dtype=self.t.int64,device=x.device);self.call(self.get,self.run,[self.desc(x,r),1,False,self.desc(value,r),self.desc(index,r)],r,x.device);return index,value
