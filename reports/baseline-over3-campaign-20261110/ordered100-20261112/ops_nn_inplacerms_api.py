"""Original generated InplaceAddRmsNorm public API and frozen run_inplace AST."""
import ctypes as C,json,hashlib,os,ast
from pathlib import Path
FROZEN_ADAPTER_SHA256='3ab7708f1d8cf953b8bbbf148fe110d4493d3f882ae28c971c49afd5f5fc2cf9'
class InplaceRmsAPI:
 def __init__(self,torch,workspace,manifest,qualification=None):
  self.t=torch;self.W=Path(workspace);self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-inplacerms-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so');self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int
  self.lib.aclnnInplaceAddRmsNormGetWorkspaceSize.argtypes=[C.c_void_p]*3+[C.c_double,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnInplaceAddRmsNormGetWorkspaceSize.restype=C.c_int;self.lib.aclnnInplaceAddRmsNorm.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnInplaceAddRmsNorm.restype=C.c_int
  self.source=Path(manifest).parent/'registered_inplace_adapter.py';assert hashlib.sha256(self.source.read_bytes()).hexdigest()==FROZEN_ADAPTER_SHA256;self.tree=ast.parse(self.source.read_text());self.run_ASTs={}
 def call(self,op,inputs,outputs,attrs):
  t=self.t;assert op=='InplaceAddRmsNorm' and len(inputs)==len(outputs)==3 and set(attrs)=={'epsilon'};x,residual,weight=inputs;y,stat,summed=outputs;assert y is x and summed is residual and x.shape==residual.shape and weight.shape==(x.shape[-1],);assert x.dtype==residual.dtype==weight.dtype and x.dtype in (t.float16,t.bfloat16);assert stat.dtype==t.float32 and tuple(stat.shape)==(*x.shape[:-1],1);assert all(a.device.type=='npu' and a.device==x.device and a.is_contiguous() for a in [x,residual,weight,stat]);keep=[x,residual,weight,stat];owned=[]
  def tensor(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);dtype={t.float16:1,t.bfloat16:27,t.float32:0}[a.dtype];p=self.base.aclCreateTensor(dims,a.ndim,dtype,strides,0,2,dims,a.ndim,a.data_ptr());assert p;owned.append(p);return p
  try:
   args=[tensor(x),tensor(residual),tensor(weight),float(attrs['epsilon']),tensor(stat)];size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnInplaceAddRmsNormGetWorkspaceSize(*args,C.byref(size),C.byref(exe));assert rc==0,('InplaceAddRmsNorm','GetWorkspaceSize',rc);ws=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(ws);rc=self.lib.aclnnInplaceAddRmsNorm(ws.data_ptr() if ws is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('InplaceAddRmsNorm','run',rc);t.npu.synchronize()
  finally:
   t.npu.synchronize()
   for p in reversed(owned):assert self.base.aclDestroyTensor(p)==0
 def prepare(self,case):
  assert case._call.signature.name=='FusedAddRMSNormFwdOp';run=next(n for n in ast.walk(self.tree) if isinstance(n,ast.FunctionDef) and n.name=='run_inplace');self.run_ASTs={'FusedAddRMSNormFwdOp':hashlib.sha256(ast.dump(run,include_attributes=False).encode()).hexdigest()};ns={'torch':self.t,'b':self,'p':case.params};mod=ast.Module(body=[run],type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,str(self.source),'exec'),ns);return ns['run_inplace']
