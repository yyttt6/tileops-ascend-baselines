"""Complete original public Matmul API, FP32 KEEP_DTYPE; exact historical037f902 timed mm/run AST."""
import ctypes as C,os,json,hashlib,ast
from pathlib import Path
class GemmPad4API:
 def __init__(self,torch,manifest,qualification=None):
  self.t=torch;self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root']);self.N=Path(manifest).parent
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-gemm-pad4-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so');self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int;self.lib.aclnnMatmulGetWorkspaceSize.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int8,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnMatmulGetWorkspaceSize.restype=C.c_int;self.lib.aclnnMatmul.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnMatmul.restype=C.c_int
 def call(self,op,inputs,outputs,attrs):
  t=self.t;assert op=='MatMulV3' and len(inputs)==4 and inputs[2:] ==[None,None] and len(outputs)==1 and attrs=={'transpose_x1':False,'transpose_x2':True};a,wnt=inputs[:2];out=outputs[0];assert a.dtype==wnt.dtype==out.dtype and a.dtype in (t.float16,t.bfloat16) and a.ndim==wnt.ndim==out.ndim==2 and a.shape[1]==wnt.shape[1] and tuple(out.shape)==(a.shape[0],wnt.shape[0]) and all(x.device.type=='npu' and x.is_contiguous() for x in [a,wnt,out]);rhs=wnt.T;keep=[a,wnt,rhs,out];owned=[]
  def tensor(view,storage_shape):
   dims=(C.c_int64*2)(*view.shape);strides=(C.c_int64*2)(*view.stride());storage=(C.c_int64*2)(*storage_shape);keep.extend([dims,strides,storage]);p=self.base.aclCreateTensor(dims,2,{t.float16:1,t.bfloat16:27}[view.dtype],strides,0,2,storage,2,view.data_ptr());assert p;owned.append(p);return p
  try:
   lhs=tensor(a,a.shape);right=tensor(rhs,wnt.shape);dst=tensor(out,out.shape);size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnMatmulGetWorkspaceSize(lhs,right,dst,0,C.byref(size),C.byref(exe));assert rc==0,('Matmul','GetWorkspaceSize','KEEP_DTYPE0_originalF16BF16',rc);ws=t.empty(size.value,dtype=t.uint8,device=a.device) if size.value else None;keep.append(ws);rc=self.lib.aclnnMatmul(ws.data_ptr() if ws is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('Matmul','run',rc);t.npu.synchronize()
  finally:
   t.npu.synchronize()
   for p in reversed(owned):assert self.base.aclDestroyTensor(p)==0
 def prepare(self,case):
  assert case._call.signature.name=='GemmFwdOp'
  from benchmarks.ascend_baselines import UnsupportedCase
  layout_source=self.N/'original-ascend_gemm_cases.py';factory_source=self.N/'original-ascend_nn_aligned_gemm.py'
  expected={'original-ascend_gemm_cases.py':'174b54193f55b2b32e7500fe2986abfc1978c8e5e9ce8e72d1367179810a0453','original-ascend_nn_aligned_gemm.py':'f13d335237964ad5ac39dae09d0e8a9d5781a36db7fe3518c3470d271009b778'}
  for p in [layout_source,factory_source]:assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[p.name]
  tree=ast.parse(layout_source.read_text());layout=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_layout');ns={'UnsupportedCase':UnsupportedCase};exec(compile(ast.fix_missing_locations(ast.Module(body=[layout],type_ignores=[])),str(layout_source),'exec'),ns)
  rows,columns,_,pack=ns['_layout'](case,(1024,256,64))
  tree=ast.parse(factory_source.read_text());factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='opsnn_aligned_gemm');run=next(n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name=='run');self.original_ASTs={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in [layout,run]}
  ns={'torch':self.t,'binding':self,'pack':pack,'rows':rows,'columns':columns};exec(compile(ast.fix_missing_locations(ast.Module(body=[run],type_ignores=[])),str(factory_source),'exec'),ns);return ns['run']
