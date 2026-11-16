"""Two complete pinned ops-math public APIs; no tensor predicate/reduction fallback."""
import ctypes as C,json,hashlib,os
from pathlib import Path
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-countnz-v1')
class CountNZAPI:
 def __init__(self,torch,dim=-1,shape=None):
  self.t=torch;self.dim=dim;self.shape=tuple(shape) if shape is not None else None;self.pending=[];self.codes={torch.float32:0,torch.float16:1,torch.bfloat16:27,torch.bool:12,torch.int64:9,torch.int32:3,torch.uint8:4};self.m=json.loads((N/'COUNTNZ_PACKAGE.json').read_text());self.root=Path(self.m['vendor'])
  for f,r in self.m['files'].items():assert hashlib.sha256((N/'private-v1'/f).read_bytes()).hexdigest()==r['sha256'],f
  assert self.root.resolve() in {Path(p).resolve() for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if p}
  self.lib=C.CDLL(str(self.root/'op_api/lib/libcust_opapi.so'),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so')
  self.neget=self.lib.aclnnNeTensorGetWorkspaceSize;self.sumget=self.lib.aclnnReduceSumGetWorkspaceSize;self.ne=self.lib.aclnnNeTensor;self.sum=self.lib.aclnnReduceSum
  self.neget.argtypes=[C.c_void_p]*3+[C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.sumget.argtypes=[C.c_void_p,C.c_void_p,C.c_bool,C.c_int,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)]
  for f in [self.neget,self.sumget,self.ne,self.sum]:f.restype=C.c_int
  for f in [self.ne,self.sum]:f.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p]
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int
  self.base.aclCreateIntArray.argtypes=[C.POINTER(C.c_int64),C.c_uint64];self.base.aclCreateIntArray.restype=C.c_void_p;self.base.aclDestroyIntArray.argtypes=[C.c_void_p];self.base.aclDestroyIntArray.restype=C.c_int
 def check(self,r,name):
  if r:raise RuntimeError(('full_CountNZ',name,r))
 def desc(self,t,r):
  d=(C.c_int64*t.ndim)(*t.shape);s=(C.c_int64*t.ndim)(*t.stride());r['keep'].extend([t,d,s]);p=self.base.aclCreateTensor(d,t.ndim,self.codes[t.dtype],s,0,2,d,t.ndim,t.data_ptr());assert p;r['tensors'].append(p);return p
 def call(self,get,run,args,r,device):
  size=C.c_uint64();exe=C.c_void_p();self.check(get(*args,C.byref(size),C.byref(exe)),'GetWorkspaceSize');ws=self.t.empty(size.value,dtype=self.t.uint8,device=device) if size.value else None;r['keep'].extend([size,exe,ws]);self.check(run(ws.data_ptr() if ws is not None else None,size.value,exe,self.t.npu.current_stream().npu_stream),'run')
 def __call__(self,x):
  assert x.ndim==2 and self.dim==-1 and x.dtype in (self.t.bool,self.t.float16,self.t.bfloat16,self.t.float32);r={'keep':[],'tensors':[],'arrays':[]};self.pending.append(r)
  x=x.contiguous();zero=self.t.zeros((),dtype=x.dtype,device=x.device);pred=self.t.empty(x.shape,dtype=self.t.bool,device=x.device)
  self.call(self.neget,self.ne,[self.desc(x,r),self.desc(zero,r),self.desc(pred,r)],r,x.device)
  dim=self.dim%x.ndim;dims=(C.c_int64*1)(dim);array=self.base.aclCreateIntArray(dims,1);assert array;r['keep'].append(dims);r['arrays'].append(array)
  shape=tuple(s for i,s in enumerate(x.shape) if i!=dim);assert self.shape is None or shape==self.shape;out=self.t.empty(shape,dtype=self.t.int64,device=x.device)
  self.call(self.sumget,self.sum,[self.desc(pred,r),array,False,self.codes[self.t.int64],self.desc(out,r)],r,x.device);return out
 def cleanup(self):
  self.t.npu.synchronize()
  while self.pending:
   r=self.pending.pop()
   for p in r['tensors']:self.check(self.base.aclDestroyTensor(p),'aclDestroyTensor')
   for p in r['arrays']:self.check(self.base.aclDestroyIntArray(p),'aclDestroyIntArray')
