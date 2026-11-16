"""Complete original FP32 aclnnMean; half widening/cast belongs to original pool wrapper."""
import ctypes as C,json,hashlib,os
from pathlib import Path
from ops_math_l1_api import MathL1API
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-pool-mean-v1')
class MeanAPI(MathL1API):
 def __init__(self,torch,dim=-1,shape=None,outdtype=None):
  assert dim==-1 and outdtype in (None,torch.float32);self.t=torch;self.shape=tuple(shape) if shape is not None else None;self.pending=[];self.codes={torch.float32:0};mfile=N/'MATH_POOL_MEAN_PACKAGE.json';m=json.loads(mfile.read_text());assert m['pin']=='64322f6487162c0934ec0364af57ed2f51175a03';root=Path(m['vendor']);self.sources={str(mfile):hashlib.sha256(mfile.read_bytes()).hexdigest()}
  for k,r in m['files'].items():
   p=Path(m['package'])/k;assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];self.sources[str(p)]=r['sha256']
  assert root.resolve() in {Path(p).resolve() for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if p};self.lib=C.CDLL(str(root/'op_api/lib/libcust_opapi.so'),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so');self.mean_get=self.lib.aclnnMeanGetWorkspaceSize;self.mean_run=self.lib.aclnnMean;self.mean_get.argtypes=[C.c_void_p,C.c_void_p,C.c_bool,C.c_int,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.mean_get.restype=C.c_int;self.mean_run.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.mean_run.restype=C.c_int
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int;self.base.aclCreateIntArray.argtypes=[C.POINTER(C.c_int64),C.c_uint64];self.base.aclCreateIntArray.restype=C.c_void_p;self.base.aclDestroyIntArray.argtypes=[C.c_void_p];self.base.aclDestroyIntArray.restype=C.c_int
 def check(self,r,name):
  if r:raise RuntimeError(('full_MathMean',name,r))
 def __call__(self,x):
  assert x.ndim==2 and x.dtype==self.t.float32;r={'keep':[],'tensors':[],'arrays':[]};self.pending.append(r);x=x.contiguous();dims=(C.c_int64*1)(1);array=self.base.aclCreateIntArray(dims,1);assert array;r['keep'].append(dims);r['arrays'].append(array);out=self.t.empty((x.shape[0],),dtype=self.t.float32,device=x.device);assert self.shape is None or tuple(out.shape)==self.shape;self.call(self.mean_get,self.mean_run,[self.desc(x,r),array,False,self.codes[out.dtype],self.desc(out,r)],r,x.device);return out
