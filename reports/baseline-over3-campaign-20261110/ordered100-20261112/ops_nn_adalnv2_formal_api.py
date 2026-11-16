"""Complete original AdaLayerNormV2 API, all three outputs per call."""
import ctypes as C,hashlib,json,os,ast
from pathlib import Path
class AdaLayerNormV2API:
 def __init__(self,torch,manifest,qualification):
  self.t=torch;self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root']);g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-ordered100/') and g['child']['pid']==os.getpgrp()
  qpath=Path(qualification);q=json.loads(qpath.read_text());assert q['private_original_device_kernels_verified'] and q['complete_original_AdaLayerNormV2_API_all3_outputs_verified'] and q['frozen4_original_checker_passed'] and q['package_manifest_sha256']==hashlib.sha256(Path(manifest).read_bytes()).hexdigest();n=qpath.parent;pg=n/q['probe_guard_receipt'];assert hashlib.sha256(pg.read_bytes()).hexdigest()==q['probe_guard_receipt_sha256'];oldg=json.loads(pg.read_text());assert oldg['boot_id']==g['boot_id'] and oldg['phase']=='finished' and oldg['performance_valid'] and oldg['workload_exit_code']==0
  result=n/q['probe_result'];assert hashlib.sha256(result.read_bytes()).hexdigest()==q['probe_result_sha256'];proof=json.loads(result.read_text());original=n/'ops_nn_adalnv2_api.py';assert hashlib.sha256(original.read_bytes()).hexdigest()==proof['source_sha256'][str(original)]
  def call_ast(path):return next(node for node in ast.walk(ast.parse(Path(path).read_text())) if isinstance(node,ast.FunctionDef) and node.name=='__call__')
  assert ast.dump(call_ast(__file__),include_attributes=False)==ast.dump(call_ast(original),include_attributes=False),'Only admission changed; complete native call must remain identical'
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so');self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int
  self.lib.aclnnAdaLayerNormV2GetWorkspaceSize.argtypes=[C.c_void_p]*5+[C.c_double]+[C.c_void_p]*3+[C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnAdaLayerNormV2GetWorkspaceSize.restype=C.c_int;self.lib.aclnnAdaLayerNormV2.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnAdaLayerNormV2.restype=C.c_int
 def __call__(self,x,scale,shift,eps):
  t=self.t;assert x.dtype==scale.dtype==shift.dtype==t.float32 and x.ndim==3 and x.shape[1]==1 and tuple(scale.shape)==tuple(shift.shape)==(x.shape[0],x.shape[2]);assert all(v.device==x.device and v.device.type=='npu' and v.is_contiguous() for v in [x,scale,shift]);out=t.empty_like(x);mean=t.empty((x.shape[0],x.shape[1],1),dtype=t.float32,device=x.device);rstd=t.empty_like(mean);owned=[];keep=[x,scale,shift,out,mean,rstd]
  def tensor(v):
   dims=(C.c_int64*v.ndim)(*v.shape);strides=(C.c_int64*v.ndim)(*v.stride());keep.extend([dims,strides]);p=self.base.aclCreateTensor(dims,v.ndim,0,strides,0,2,dims,v.ndim,v.data_ptr());assert p;owned.append(p);return p
  try:
   src,sc,sh,y,m,r=[tensor(v) for v in [x,scale,shift,out,mean,rstd]];size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnAdaLayerNormV2GetWorkspaceSize(src,sc,sh,None,None,float(eps),y,m,r,C.byref(size),C.byref(exe));assert rc==0,('AdaLayerNormV2','GetWorkspaceSize',rc);workspace=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(workspace);rc=self.lib.aclnnAdaLayerNormV2(workspace.data_ptr() if workspace is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('AdaLayerNormV2','run',rc);t.npu.synchronize();return out,mean,rstd
  finally:
   t.npu.synchronize()
   for p in reversed(owned):assert self.base.aclDestroyTensor(p)==0
