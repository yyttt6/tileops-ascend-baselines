"""Whole pinned public APIs. Probe exemption is confined to the actual guarded probe owner."""
import ctypes as C,hashlib,json,os
from pathlib import Path
class MiscNativeAPI:
 def __init__(self,torch,manifest,qualification=None):
  self.t=torch;self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-misc-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so')
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p
  self.base.aclCreateTensorList.argtypes=[C.POINTER(C.c_void_p),C.c_uint64];self.base.aclCreateTensorList.restype=C.c_void_p
  self.base.aclCreateScalar.argtypes=[C.c_void_p,C.c_int];self.base.aclCreateScalar.restype=C.c_void_p
  for n in ['Tensor','TensorList','Scalar']:
   f=getattr(self.base,'aclDestroy'+n);f.argtypes=[C.c_void_p];f.restype=C.c_int
  for op,count in [('ForeachPowScalar',3),('Softplus',4)]:
   g=getattr(self.lib,'aclnn'+op+'GetWorkspaceSize');g.argtypes=[C.c_void_p]*count+[C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];g.restype=C.c_int
   r=getattr(self.lib,'aclnn'+op);r.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];r.restype=C.c_int
 def __call__(self,x,op,beta=1.,threshold=20.):
  t=self.t;codes={t.float16:1,t.bfloat16:27,t.float32:0};assert op in ('ForeachPowScalar','Softplus') and x.dtype in codes and x.device.type=='npu' and x.is_contiguous();y=t.empty_like(x);keep=[x,y];owned=[]
  def tensor(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);p=self.base.aclCreateTensor(dims,a.ndim,codes[a.dtype],strides,0,2,dims,a.ndim,a.data_ptr());assert p;owned.append(('Tensor',p));return p
  def one_list(a):
   p=tensor(a);arr=(C.c_void_p*1)(p);keep.append(arr);q=self.base.aclCreateTensorList(arr,1);assert q;owned.remove(('Tensor',p));owned.append(('TensorList',q));return q
  def scalar(v):
   value=C.c_float(float(v));keep.append(value);p=self.base.aclCreateScalar(C.byref(value),0);assert p;owned.append(('Scalar',p));return p
  try:
   if op=='ForeachPowScalar':
    control=t.tensor(-.5,dtype=t.float16 if x.dtype==t.float16 else t.float32,device=x.device);keep.append(control);args=[one_list(x),tensor(control),one_list(y)]
   else:args=[tensor(x),scalar(beta),scalar(threshold),tensor(y)]
   size=C.c_uint64();exe=C.c_void_p();rc=getattr(self.lib,'aclnn'+op+'GetWorkspaceSize')(*args,C.byref(size),C.byref(exe));assert rc==0,(op,'GetWorkspaceSize',rc)
   workspace=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(workspace);rc=getattr(self.lib,'aclnn'+op)(workspace.data_ptr() if workspace is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,(op,'run',rc);t.npu.synchronize();return y
  finally:
   t.npu.synchronize()
   for kind,p in reversed(owned):assert getattr(self.base,'aclDestroy'+kind)(p)==0
