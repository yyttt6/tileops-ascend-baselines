"""Thin complete original math C APIs; no tensor arithmetic fallback."""
import ctypes as C,json,hashlib,os
from pathlib import Path
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-dequant-v1')
class MathDequantAPI:
 def __init__(self,torch,op):
  assert op=='Mul';self.t=torch;self.op=op;self.n=2;self.pending=[];self.codes={torch.float32:0,torch.float16:1,torch.bfloat16:27,torch.bool:12};self.m=json.loads((N/'DEQUANT_PACKAGE.json').read_text());self.root=Path(self.m['vendor']);self.sources={str(N/'DEQUANT_PACKAGE.json'):hashlib.sha256((N/'DEQUANT_PACKAGE.json').read_bytes()).hexdigest()};assert self.m['pin']=='64322f6487162c0934ec0364af57ed2f51175a03'
  for f,r in self.m['files'].items():
   file=N/'private-v1'/f;assert hashlib.sha256(file.read_bytes()).hexdigest()==r['sha256'],f;self.sources[str(file)]=r['sha256']
  assert self.root.resolve() in {Path(p).resolve() for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if p}
  self.lib=C.CDLL(str(self.root/'op_api/lib/libcust_opapi.so'),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so');self.get=getattr(self.lib,'aclnn'+op+'GetWorkspaceSize');self.run=getattr(self.lib,'aclnn'+op);self.get.argtypes=[C.c_void_p]*(self.n+1)+[C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.get.restype=C.c_int;self.run.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.run.restype=C.c_int
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int
 def check(self,r,name):
  if r:raise RuntimeError((self.op,name,r))
 def desc(self,t,r):
  d=(C.c_int64*t.ndim)(*t.shape);s=(C.c_int64*t.ndim)(*t.stride());r['keep'].extend([t,d,s]);p=self.base.aclCreateTensor(d,t.ndim,self.codes[t.dtype],s,0,2,d,t.ndim,t.data_ptr());assert p;r['ptrs'].append(p);return p
 def __call__(self,*values):
  assert len(values)==self.n and all(v.dtype==self.t.float32 for v in values)
  xs=[v.contiguous() for v in values];out=self.t.empty(self.t.broadcast_shapes(*(x.shape for x in xs)),dtype=self.t.float32,device=xs[0].device);r={'keep':[],'ptrs':[]};self.pending.append(r);ptrs=[self.desc(v,r) for v in [*xs,out]];size=C.c_uint64();exe=C.c_void_p();self.check(self.get(*ptrs,C.byref(size),C.byref(exe)),'GetWorkspaceSize');ws=self.t.empty(size.value,dtype=self.t.uint8,device=out.device) if size.value else None;r['keep'].extend([exe,ws]);self.check(self.run(ws.data_ptr() if ws is not None else None,size.value,exe,self.t.npu.current_stream().npu_stream),'run');self.cleanup();return out
 def cleanup(self):
  self.t.npu.synchronize()
  while self.pending:
   r=self.pending.pop()
   for p in r['ptrs']:self.check(self.base.aclDestroyTensor(p),'aclDestroyTensor')
