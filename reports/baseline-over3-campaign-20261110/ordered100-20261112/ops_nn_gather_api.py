"""Complete original publicaclnnGather -> originalGatherElements, finiteWhere layout."""
import ctypes as C,json,hashlib,os,ast
from pathlib import Path
class GatherAPI:
 def __init__(self,torch,workspace,manifest,qualification=None):
  self.t=torch;self.W=Path(workspace);self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-gather-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so');self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p;self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int
  self.lib.aclnnGatherGetWorkspaceSize.argtypes=[C.c_void_p,C.c_int64,C.c_void_p,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnGatherGetWorkspaceSize.restype=C.c_int;self.lib.aclnnGather.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnGather.restype=C.c_int
  self.source=self.W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/elementwise.py';self.tree=ast.parse(self.source.read_text());self.run_ASTs={}
 def call(self,op,inputs,outputs,attrs):
  t=self.t;assert op=='GatherElements' and len(inputs)==2 and len(outputs)==1 and attrs=={'dim':1};x,index=inputs;out=outputs[0];assert x.ndim==index.ndim==out.ndim==2 and x.shape[1]==2 and tuple(index.shape)==tuple(out.shape)==(x.shape[0],1);assert x.dtype==out.dtype and x.dtype in (t.float16,t.bfloat16,t.float32) and index.dtype==t.int32;assert all(a.device.type=='npu' and a.device==x.device and a.is_contiguous() for a in [x,index,out]);keep=[x,index,out];owned=[]
  def tensor(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);dtype={t.float16:1,t.bfloat16:27,t.float32:0,t.int32:3}[a.dtype];p=self.base.aclCreateTensor(dims,a.ndim,dtype,strides,0,2,dims,a.ndim,a.data_ptr());assert p;owned.append(p);return p
  try:
   args=[tensor(x),1,tensor(index),tensor(out)];size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnGatherGetWorkspaceSize(*args,C.byref(size),C.byref(exe));assert rc==0,('Gather','GetWorkspaceSize',rc);ws=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(ws);rc=self.lib.aclnnGather(ws.data_ptr() if ws is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('Gather','run',rc);t.npu.synchronize()
  finally:
   t.npu.synchronize()
   for p in reversed(owned):assert self.base.aclDestroyTensor(p)==0
 def prepare(self,case):
  name=case._call.signature.name;assert name=='WhereFwdOp';outer=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='where');run=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='run');self.run_ASTs[name]=hashlib.sha256(ast.dump(run,include_attributes=False).encode()).hexdigest();ns={'torch':self.t,'binding':self};mod=ast.Module(body=[run],type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,str(self.source),'exec'),ns);return ns['run']
