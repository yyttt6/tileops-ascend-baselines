"""Complete original AdaptiveAvgPool2d API; declared1x1 ReduceMean SDK dependency."""
import ctypes as C,hashlib,json,os,ast,math
from pathlib import Path
class PoolMeanAPI:
 def __init__(self,torch,workspace,manifest,qualification=None):
  self.t=torch;self.W=Path(workspace);self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-poolmean-') and g['child']['pid']==os.getpgrp()
  else:
   q=json.loads(Path(qualification).read_text());assert q['complete_original_api_declared_dependencies_verified'] and q['native_input_dtype']=='torch.float32'
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so')
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p
  self.base.aclCreateIntArray.argtypes=[C.POINTER(C.c_int64),C.c_uint64];self.base.aclCreateIntArray.restype=C.c_void_p
  for kind in ['Tensor','IntArray']:
   f=getattr(self.base,'aclDestroy'+kind);f.argtypes=[C.c_void_p];f.restype=C.c_int
  self.lib.aclnnAdaptiveAvgPool2dGetWorkspaceSize.argtypes=[C.c_void_p]*3+[C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnAdaptiveAvgPool2dGetWorkspaceSize.restype=C.c_int
  self.lib.aclnnAdaptiveAvgPool2d.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnAdaptiveAvgPool2d.restype=C.c_int
  self.source=self.W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/pooling.py';self.tree=ast.parse(self.source.read_text());self.run_ASTs={}
 def call(self,op,inputs,outputs,attrs):
  t=self.t;assert op=='AdaptiveAvgPool2d' and len(inputs)==len(outputs)==1 and list(attrs['output_size'])==[1,1];x=inputs[0];out=outputs[0];assert x.dtype==out.dtype==t.float32 and x.device.type=='npu' and x.is_contiguous() and out.is_contiguous() and x.ndim in (3,4)
  keep=[x,out];owned=[]
  def tensor(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);p=self.base.aclCreateTensor(dims,a.ndim,0,strides,0,47 if a.ndim==3 else 0,dims,a.ndim,a.data_ptr());assert p;owned.append(('Tensor',p));return p
  try:
   size_data=(C.c_int64*2)(1,1);keep.append(size_data);p=self.base.aclCreateIntArray(size_data,2);assert p;owned.append(('IntArray',p));src=tensor(x);dst=tensor(out);size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnAdaptiveAvgPool2dGetWorkspaceSize(src,p,dst,C.byref(size),C.byref(exe));assert rc==0,('AdaptiveAvgPool2d','GetWorkspaceSize',rc)
   ws=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(ws);rc=self.lib.aclnnAdaptiveAvgPool2d(ws.data_ptr() if ws is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('AdaptiveAvgPool2d','run',rc);t.npu.synchronize()
  finally:
   t.npu.synchronize()
   for kind,p in reversed(owned):assert getattr(self.base,'aclDestroy'+kind)(p)==0
 def prepare(self,case):
  name=case._call.signature.name;mapping={'AdaptiveAvgPool2dFwdOp':'pool','MeanFwdOp':'reduction_pool','MeanPoolingFwdOp':'chunk_mean'};assert name in mapping;p=case.params
  if name=='AdaptiveAvgPool2dFwdOp':assert list(p['output_size'])==[1,1]
  outer=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name==mapping[name]);run=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='run');self.run_ASTs[name]=hashlib.sha256(ast.dump(run,include_attributes=False).encode()).hexdigest();ns={'torch':self.t,'prod':math.prod,'binding':self,'case':case,'p':p,'name':name,'api':'AdaptiveAvgPool2d','out_shape':tuple(p['output'][0]),'indices_requested':False};tree=ast.Module(body=[run],type_ignores=[]);ast.fix_missing_locations(tree);exec(compile(tree,str(self.source),'exec'),ns);return ns['run']
