"""Complete original LayerNorm API, with all mean/rstd outputs computed per call."""
import ctypes as C,hashlib,json,os
from pathlib import Path
class LayerNormNativeAPI:
 def __init__(self,torch,manifest,qualification=None):
  self.t=torch;self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-adaln-decode-ln-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so')
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p
  self.base.aclCreateIntArray.argtypes=[C.POINTER(C.c_int64),C.c_uint64];self.base.aclCreateIntArray.restype=C.c_void_p
  for kind in ['Tensor','IntArray']:
   f=getattr(self.base,'aclDestroy'+kind);f.argtypes=[C.c_void_p];f.restype=C.c_int
  self.lib.aclnnLayerNormGetWorkspaceSize.argtypes=[C.c_void_p]*4+[C.c_double]+[C.c_void_p]*3+[C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnLayerNormGetWorkspaceSize.restype=C.c_int
  self.lib.aclnnLayerNorm.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnLayerNorm.restype=C.c_int
 def __call__(self,x,weight,bias,normalized_shape,eps):
  t=self.t;codes={t.float16:1,t.bfloat16:27,t.float32:0};shape=tuple(normalized_shape);assert x.dtype==t.float32 and x.device.type=='npu' and x.is_contiguous() and shape and tuple(x.shape[-len(shape):])==shape
  assert weight is not None and bias is not None and all(v.dtype==x.dtype and tuple(v.shape)==shape and v.device==x.device and v.is_contiguous() for v in [weight,bias])
  out=t.empty_like(x);stats_shape=tuple(x.shape[:-len(shape)])+(1,)*len(shape);mean=t.empty(stats_shape,dtype=t.float32,device=x.device);rstd=t.empty_like(mean);keep=[x,weight,bias,out,mean,rstd];owned=[]
  def tensor(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);p=self.base.aclCreateTensor(dims,a.ndim,codes[a.dtype],strides,0,2,dims,a.ndim,a.data_ptr());assert p;owned.append(('Tensor',p));return p
  try:
   dims=(C.c_int64*len(shape))(*shape);keep.append(dims);norm=self.base.aclCreateIntArray(dims,len(shape));assert norm;owned.append(('IntArray',norm));src=tensor(x);w=tensor(weight);b=tensor(bias);y=tensor(out);m=tensor(mean);r=tensor(rstd);size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnLayerNormGetWorkspaceSize(src,norm,w,b,float(eps),y,m,r,C.byref(size),C.byref(exe));assert rc==0,('LayerNorm','GetWorkspaceSize',rc)
   workspace=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(workspace);rc=self.lib.aclnnLayerNorm(workspace.data_ptr() if workspace is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('LayerNorm','run',rc);t.npu.synchronize();return out,mean,rstd
  finally:
   t.npu.synchronize()
   for kind,p in reversed(owned):assert getattr(self.base,'aclDestroy'+kind)(p)==0
