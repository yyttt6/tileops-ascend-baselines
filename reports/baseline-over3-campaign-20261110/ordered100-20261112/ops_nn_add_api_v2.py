"""Complete original ForeachAddListV2, a whole existing-library API for Add."""
import ctypes as C,hashlib,json,os,struct
from pathlib import Path
class AddNativeAPI:
 def __init__(self,torch,manifest,qualification=None):
  self.t=torch;self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-add-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so')
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p
  self.base.aclCreateTensorList.argtypes=[C.POINTER(C.c_void_p),C.c_uint64];self.base.aclCreateTensorList.restype=C.c_void_p
  self.base.aclCreateScalar.argtypes=[C.c_void_p,C.c_int];self.base.aclCreateScalar.restype=C.c_void_p
  for name in ['Tensor','TensorList','Scalar']:
   f=getattr(self.base,'aclDestroy'+name);f.argtypes=[C.c_void_p];f.restype=C.c_int
  self.lib.aclnnForeachAddListV2GetWorkspaceSize.argtypes=[C.c_void_p]*4+[C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnForeachAddListV2GetWorkspaceSize.restype=C.c_int
  self.lib.aclnnForeachAddListV2.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnForeachAddListV2.restype=C.c_int
 def __call__(self,x,y,alpha=1.):
  t=self.t;codes={t.float16:1,t.bfloat16:27,t.float32:0};assert x.dtype==y.dtype and x.dtype in codes and x.device==y.device and x.device.type=='npu' and x.shape==y.shape and x.is_contiguous() and y.is_contiguous();out=t.empty_like(x);keep=[x,y,out];owned=[]
  def one_list(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);p=self.base.aclCreateTensor(dims,a.ndim,codes[a.dtype],strides,0,2,dims,a.ndim,a.data_ptr());assert p;owned.append(('Tensor',p));arr=(C.c_void_p*1)(p);keep.append(arr);q=self.base.aclCreateTensorList(arr,1);assert q;owned.remove(('Tensor',p));owned.append(('TensorList',q));return q
  try:
   left=one_list(x);right=one_list(y);value=C.c_uint16(int.from_bytes(struct.pack('<e',float(alpha)),'little')) if x.dtype==t.float16 else C.c_float(float(alpha));keep.append(value);scalar=self.base.aclCreateScalar(C.byref(value),1 if x.dtype==t.float16 else 0);assert scalar;owned.append(('Scalar',scalar));dest=one_list(out);size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnForeachAddListV2GetWorkspaceSize(left,right,scalar,dest,C.byref(size),C.byref(exe));assert rc==0,('ForeachAddListV2','GetWorkspaceSize',rc)
   workspace=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(workspace);rc=self.lib.aclnnForeachAddListV2(workspace.data_ptr() if workspace is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('ForeachAddListV2','run',rc);t.npu.synchronize();return out
  finally:
   t.npu.synchronize()
   for kind,p in reversed(owned):assert getattr(self.base,'aclDestroy'+kind)(p)==0
