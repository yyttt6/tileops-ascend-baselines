"""Samepin complete public MaxPool3dWithArgmax CAPI, exact original pooling wrapper AST."""
import ctypes as C,json,hashlib,os,ast,math
from pathlib import Path
class MaxPoolAPI:
 def __init__(self,torch,workspace,manifest,qualification=None):
  self.t=torch;self.W=Path(workspace);self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-maxpool-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so')
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclCreateIntArray.argtypes=[C.POINTER(C.c_int64),C.c_uint64];self.base.aclCreateIntArray.restype=C.c_void_p
  for kind in ['Tensor','IntArray']:f=getattr(self.base,'aclDestroy'+kind);f.argtypes=[C.c_void_p];f.restype=C.c_int
  self.lib.aclnnMaxPool3dWithArgmaxGetWorkspaceSize.argtypes=[C.c_void_p]*5+[C.c_bool,C.c_void_p,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnMaxPool3dWithArgmaxGetWorkspaceSize.restype=C.c_int;self.lib.aclnnMaxPool3dWithArgmax.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnMaxPool3dWithArgmax.restype=C.c_int
  self.source=self.W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/pooling.py';self.tree=ast.parse(self.source.read_text());self.run_ASTs={}
 def call(self,op,inputs,outputs,attrs,formats=None):
  t=self.t;assert op=='MaxPool3DWithArgmax' and len(inputs)==1 and len(outputs)==2 and formats==[30,30,30];x=inputs[0];out,idx=outputs;assert x.ndim==out.ndim==idx.ndim==5 and x.dtype==out.dtype and x.dtype in (t.float16,t.bfloat16,t.float32) and idx.dtype==t.int64;assert attrs['data_format']=='NCDHW' and attrs['dtype']==9;assert all(a.device.type=='npu' and a.is_contiguous() for a in [x,out,idx]);keep=[x,out,idx];owned=[]
  def tensor(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);dtype={t.float16:1,t.bfloat16:27,t.float32:0,t.int64:9}[a.dtype];p=self.base.aclCreateTensor(dims,a.ndim,dtype,strides,0,30,dims,a.ndim,a.data_ptr());assert p;owned.append(('Tensor',p));return p
  def ints(values):
   data=(C.c_int64*len(values))(*values);keep.append(data);p=self.base.aclCreateIntArray(data,len(values));assert p;owned.append(('IntArray',p));return p
  try:
   assert len(attrs['ksize'])==len(attrs['strides'])==3;assert len(attrs['pads'])==len(attrs['dilation'])==3
   args=[tensor(x),ints(attrs['ksize']),ints(attrs['strides']),ints(attrs['pads']),ints(attrs['dilation']),bool(attrs['ceil_mode']),tensor(out),tensor(idx)];size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnMaxPool3dWithArgmaxGetWorkspaceSize(*args,C.byref(size),C.byref(exe));assert rc==0,('MaxPool3dWithArgmax','GetWorkspaceSize',rc);ws=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(ws);rc=self.lib.aclnnMaxPool3dWithArgmax(ws.data_ptr() if ws is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('MaxPool3dWithArgmax','run',rc);t.npu.synchronize()
  finally:
   t.npu.synchronize()
   for kind,p in reversed(owned):assert getattr(self.base,'aclDestroy'+kind)(p)==0
 def prepare(self,case):
  name=case._call.signature.name;assert name in ('MaxPool1dFwdOp','MaxPool1dIndicesFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp');p=case.params;outer=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='pool');run=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='run');tuplefn=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tuple');self.run_ASTs[name]=hashlib.sha256(ast.dump(run,include_attributes=False).encode()).hexdigest();ns={'torch':self.t,'binding':self,'case':case,'p':p,'name':name,'api':'MaxPool3DWithArgmax','out_shape':tuple(p['output' if 'output' in p else 'values'][0]),'indices_requested':'Indices' in name};tree=ast.Module(body=[tuplefn,run],type_ignores=[]);ast.fix_missing_locations(tree);exec(compile(tree,str(self.source),'exec'),ns);return ns['run']
