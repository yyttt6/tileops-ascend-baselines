"""Full original aclnnNorm p0 API; exact count_nonzero and truth01 ASTs."""
import ctypes as C,json,hashlib,os,ast
from pathlib import Path
class LpNormAPI:
 def __init__(self,torch,workspace,manifest,qualification=None):
  self.t=torch;self.W=Path(workspace);self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-lpnorm-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so')
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p
  self.base.aclCreateIntArray.argtypes=[C.POINTER(C.c_int64),C.c_uint64];self.base.aclCreateIntArray.restype=C.c_void_p;self.base.aclCreateScalar.argtypes=[C.c_void_p,C.c_int];self.base.aclCreateScalar.restype=C.c_void_p
  for kind in ['Tensor','IntArray','Scalar']:
   f=getattr(self.base,'aclDestroy'+kind);f.argtypes=[C.c_void_p];f.restype=C.c_int
  self.lib.aclnnNormGetWorkspaceSize.argtypes=[C.c_void_p]*3+[C.c_bool,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnNormGetWorkspaceSize.restype=C.c_int;self.lib.aclnnNorm.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnNorm.restype=C.c_int
  self.source=self.W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/normalization.py';self.truth_source=self.source.with_name('truth.py');self.tree=ast.parse(self.source.read_text());self.run_ASTs={}
 def call(self,op,inputs,outputs,attrs):
  t=self.t;assert op=='LpNormV2' and len(inputs)==len(outputs)==1 and set(attrs)=={'p','axes','keepdim','epsilon'};assert attrs['p']==0.0 and attrs['epsilon']==0.0 and attrs['keepdim'] is False;x=inputs[0];out=outputs[0];assert x.dtype==out.dtype==t.float32 and x.device.type=='npu' and out.device==x.device and x.is_contiguous() and out.is_contiguous();keep=[x,out];owned=[]
  def tensor(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);p=self.base.aclCreateTensor(dims,a.ndim,0,strides,0,2,dims,a.ndim,a.data_ptr());assert p;owned.append(('Tensor',p));return p
  try:
   axes=list(attrs['axes']);data=(C.c_int64*len(axes))(*axes);value=C.c_float(attrs['p']);keep.extend([data,value]);dim=self.base.aclCreateIntArray(data,len(axes));assert dim;owned.append(('IntArray',dim));p=self.base.aclCreateScalar(C.byref(value),0);assert p;owned.append(('Scalar',p));src=tensor(x);dst=tensor(out);size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnNormGetWorkspaceSize(src,p,dim,attrs['keepdim'],dst,C.byref(size),C.byref(exe));assert rc==0,('Norm','GetWorkspaceSize',rc);ws=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(ws);rc=self.lib.aclnnNorm(ws.data_ptr() if ws is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('Norm','run',rc);t.npu.synchronize()
  finally:
   t.npu.synchronize()
   for kind,p in reversed(owned):assert getattr(self.base,'aclDestroy'+kind)(p)==0
 def prepare(self,case):
  from benchmarks.ascend_baselines import UnsupportedCase
  assert case._call.signature.name=='CountNonzeroFwdOp';outer=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='count_nonzero');run=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='run');truth=next(n for n in ast.parse(self.truth_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='truth01');self.run_ASTs={'CountNonzeroFwdOp':hashlib.sha256(ast.dump(run,include_attributes=False).encode()).hexdigest(),'truth01':hashlib.sha256(ast.dump(truth,include_attributes=False).encode()).hexdigest()};ns={'torch':self.t,'binding':self,'case':case,'p':case.params,'UnsupportedCase':UnsupportedCase};mod=ast.Module(body=[truth,run],type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,str(self.source),'exec'),ns);return ns['run']
