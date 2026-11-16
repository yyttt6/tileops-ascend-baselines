"""Original public AdaptiveMaxPool3d CAPI; finite global output1 scope only."""
import ctypes as C,json,hashlib,os,ast
from pathlib import Path
class AdaptiveMaxPoolAPI:
 def __init__(self,torch,workspace,manifest,qualification=None):
  self.t=torch;self.W=Path(workspace);self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-adaptivemaxpool-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so')
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclCreateIntArray.argtypes=[C.POINTER(C.c_int64),C.c_uint64];self.base.aclCreateIntArray.restype=C.c_void_p
  for kind in ['Tensor','IntArray']:f=getattr(self.base,'aclDestroy'+kind);f.argtypes=[C.c_void_p];f.restype=C.c_int
  self.lib.aclnnAdaptiveMaxPool3dGetWorkspaceSize.argtypes=[C.c_void_p]*4+[C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnAdaptiveMaxPool3dGetWorkspaceSize.restype=C.c_int;self.lib.aclnnAdaptiveMaxPool3d.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnAdaptiveMaxPool3d.restype=C.c_int
  self.source=self.W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/adaptive_max_lift.py';self.tree=ast.parse(self.source.read_text());self.run_ASTs={}
 def call(self,op,inputs,outputs,attrs):
  t=self.t;assert op=='AdaptiveMaxPool3d' and len(inputs)==1 and len(outputs)==2 and attrs=={'output_size':[1,1,1],'indices_dtype':3};x=inputs[0];out,idx=outputs;assert x.ndim==out.ndim==idx.ndim==5 and x.dtype==out.dtype and x.dtype in (t.float16,t.bfloat16,t.float32) and idx.dtype==t.int32;assert tuple(out.shape)==tuple(x.shape[:2])+(1,1,1) and idx.shape==out.shape;assert all(a.device.type=='npu' and a.is_contiguous() for a in [x,out,idx]);keep=[x,out,idx];owned=[]
  def tensor(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);dtype={t.float16:1,t.bfloat16:27,t.float32:0,t.int32:3}[a.dtype];p=self.base.aclCreateTensor(dims,a.ndim,dtype,strides,0,30,dims,a.ndim,a.data_ptr());assert p;owned.append(('Tensor',p));return p
  def ints(values):
   data=(C.c_int64*len(values))(*values);keep.append(data);p=self.base.aclCreateIntArray(data,len(values));assert p;owned.append(('IntArray',p));return p
  try:
   args=[tensor(x),ints(attrs['output_size']),tensor(out),tensor(idx)];size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnAdaptiveMaxPool3dGetWorkspaceSize(*args,C.byref(size),C.byref(exe));assert rc==0,('AdaptiveMaxPool3d','GetWorkspaceSize',rc);ws=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(ws);rc=self.lib.aclnnAdaptiveMaxPool3d(ws.data_ptr() if ws is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('AdaptiveMaxPool3d','run',rc);t.npu.synchronize()
  finally:
   t.npu.synchronize()
   for kind,p in reversed(owned):assert getattr(self.base,'aclDestroy'+kind)(p)==0
 def prepare(self,case):
  name=case._call.signature.name;assert name in ('AdaptiveMaxPool2dFwdOp','AdaptiveMaxPool2dIndicesFwdOp');p=case.params;size=p['output_size'];assert size==1 or tuple(size)==(1,1);outer=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='factory');run=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='run');self.run_ASTs[name]=hashlib.sha256(ast.dump(run,include_attributes=False).encode()).hexdigest();outkey='output' if 'output' in p else 'values';ns={'torch':self.t,'binding':self,'p':p,'out_shape':tuple(p[outkey][0]),'indices_requested':'Indices' in name};tree=ast.Module(body=[run],type_ignores=[]);ast.fix_missing_locations(tree);exec(compile(tree,str(self.source),'exec'),ns);return ns['run']

 def prepare_reduction(self,case):
  from math import prod
  name=case._call.signature.name;assert name=='ArgmaxFwdOp';p=case.params;source=self.W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/pooling.py';tree=ast.parse(source.read_text());outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reduction_pool');run=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='run');self.run_ASTs[name]=hashlib.sha256(ast.dump(run,include_attributes=False).encode()).hexdigest();ns={'torch':self.t,'binding':self,'p':p,'name':name,'prod':prod,'out_shape':tuple(p['output'][0]),'api':'AdaptiveMaxPool3d'};mod=ast.Module(body=[run],type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,str(source),'exec'),ns);return ns['run']
