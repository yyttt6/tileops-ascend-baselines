"""Full original DynamicBlockQuant v1 API and original opsnn_fp8_cases timed AST."""
import ctypes as C,json,hashlib,os,ast
from pathlib import Path
class FP8QuantAPI:
 def __init__(self,torch,workspace,manifest,qualification=None):
  self.t=torch;self.W=Path(workspace);self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-fp8quant-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so');self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int
  self.lib.aclnnDynamicBlockQuantGetWorkspaceSize.argtypes=[C.c_void_p,C.c_double,C.c_char_p,C.c_int64,C.c_int64,C.c_int64,C.c_void_p,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnDynamicBlockQuantGetWorkspaceSize.restype=C.c_int;self.lib.aclnnDynamicBlockQuant.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnDynamicBlockQuant.restype=C.c_int
  self.source=self.W/'TileOPs/benchmarks/ascend_quant_cases.py';self.tree=ast.parse(self.source.read_text());self.run_ASTs={}
 def call(self,op,inputs,outputs,attrs):
  t=self.t;assert op=='DynamicBlockQuant' and len(inputs)==1 and len(outputs)==2 and set(attrs)=={'min_scale','round_mode','dst_type','row_block_size','col_block_size'};assert attrs['min_scale']==0.0 and attrs['round_mode']=='rint' and attrs['dst_type']==36 and attrs['row_block_size']==1 and attrs['col_block_size'] in (128,256);x=inputs[0];codes,scales=outputs;assert x.ndim==2 and x.shape[1]==attrs['col_block_size'] and x.dtype==scales.dtype==t.float32 and codes.dtype==t.float8_e4m3fn and codes.shape==x.shape and scales.shape==(x.shape[0],1);assert all(a.device.type=='npu' and a.device==x.device and a.is_contiguous() for a in [x,codes,scales]);keep=[x,codes,scales];owned=[]
  def tensor(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);dtype={t.float32:0,t.float8_e4m3fn:36}[a.dtype];p=self.base.aclCreateTensor(dims,a.ndim,dtype,strides,0,2,dims,a.ndim,a.data_ptr());assert p;owned.append(p);return p
  try:
   mode=attrs['round_mode'].encode();keep.append(mode);args=[tensor(x),float(attrs['min_scale']),mode,int(attrs['dst_type']),int(attrs['row_block_size']),int(attrs['col_block_size']),tensor(codes),tensor(scales)];size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnDynamicBlockQuantGetWorkspaceSize(*args,C.byref(size),C.byref(exe));assert rc==0,('DynamicBlockQuant','GetWorkspaceSize',rc);ws=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(ws);rc=self.lib.aclnnDynamicBlockQuant(ws.data_ptr() if ws is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('DynamicBlockQuant','run',rc);t.npu.synchronize()
  finally:
   t.npu.synchronize()
   for p in reversed(owned):assert self.base.aclDestroyTensor(p)==0
 def prepare(self,case):
  assert case._call.signature.name=='FP8QuantFwdOp';hidden=case.inputs[0].shape[-1];assert hidden in (64,128);outer=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='opsnn_fp8_cases');run=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='run');self.run_ASTs={'FP8QuantFwdOp':hashlib.sha256(ast.dump(run,include_attributes=False).encode()).hexdigest()};ns={'torch':self.t,'F':self.t.nn.functional,'binding':self,'hidden':hidden,'block':2*hidden};mod=ast.Module(body=[run],type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,str(self.source),'exec'),ns);return ns['run']
