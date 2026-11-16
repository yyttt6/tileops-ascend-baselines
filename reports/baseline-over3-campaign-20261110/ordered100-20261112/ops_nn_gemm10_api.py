"""Complete original public Matmul API, FP32 KEEP_DTYPE; exact historical037f902 timed mm/run AST."""
import ctypes as C,os,json,hashlib,ast
from pathlib import Path
class Gemm10API:
 def __init__(self,torch,manifest,qualification=None):
  self.t=torch;self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root']);self.N=Path(manifest).parent
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-gemm10-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  self.source=self.N/'frozen_gemm_adapter_037f902.py';assert hashlib.sha256(self.source.read_bytes()).hexdigest()=='037f902f6779aa1d84dad1d5d04777e6754cec5e9b3d653b7339ac5f0dd200d7';self.tree=ast.parse(self.source.read_text());p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so');self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int;self.lib.aclnnMatmulGetWorkspaceSize.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int8,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnMatmulGetWorkspaceSize.restype=C.c_int;self.lib.aclnnMatmul.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnMatmul.restype=C.c_int
 def call(self,op,inputs,outputs,attrs):
  t=self.t;assert op=='MatMulV3' and len(inputs)==4 and inputs[2:] ==[None,None] and len(outputs)==1 and attrs=={'transpose_x1':False,'transpose_x2':True};a,wnt=inputs[:2];out=outputs[0];assert a.dtype==wnt.dtype==out.dtype==t.float32 and a.ndim==wnt.ndim==out.ndim==2 and a.shape[1]==wnt.shape[1] and tuple(out.shape)==(a.shape[0],wnt.shape[0]) and all(x.device.type=='npu' and x.is_contiguous() for x in [a,wnt,out]);rhs=wnt.T;keep=[a,wnt,rhs,out];owned=[]
  def tensor(view,storage_shape):
   dims=(C.c_int64*2)(*view.shape);strides=(C.c_int64*2)(*view.stride());storage=(C.c_int64*2)(*storage_shape);keep.extend([dims,strides,storage]);p=self.base.aclCreateTensor(dims,2,0,strides,0,2,storage,2,view.data_ptr());assert p;owned.append(p);return p
  try:
   lhs=tensor(a,a.shape);right=tensor(rhs,wnt.shape);dst=tensor(out,out.shape);size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnMatmulGetWorkspaceSize(lhs,right,dst,0,C.byref(size),C.byref(exe));assert rc==0,('Matmul','GetWorkspaceSize','KEEP_DTYPE0_FP32',rc);ws=t.empty(size.value,dtype=t.uint8,device=a.device) if size.value else None;keep.append(ws);rc=self.lib.aclnnMatmul(ws.data_ptr() if ws is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('Matmul','run',rc);t.npu.synchronize()
  finally:
   t.npu.synchronize()
   for p in reversed(owned):assert self.base.aclDestroyTensor(p)==0
 def prepare(self,case):
  assert case._call.signature.name=='GemmFwdOp';p=case.params
  branch=next(n for n in ast.walk(self.tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and ast.dump(n.test,include_attributes=False)=="Compare(left=Name(id='op', ctx=Load()), ops=[In()], comparators=[Tuple(elts=[Constant(value='GemmFwdOp'), Constant(value='GroupedGemmFwdOp')], ctx=Load())])")
  mm=next(n for n in branch.body if isinstance(n,ast.FunctionDef) and n.name=='mm');gate=next(n for n in branch.body if isinstance(n,ast.If));run=next(n for n in gate.body if isinstance(n,ast.FunctionDef) and n.name=='run');self.original_ASTs={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in [mm,run]};ns={'torch':self.t,'b':self,'p':p};tree=ast.Module(body=[mm,run],type_ignores=[]);ast.fix_missing_locations(tree);exec(compile(tree,str(self.source),'exec'),ns);return ns['run']
