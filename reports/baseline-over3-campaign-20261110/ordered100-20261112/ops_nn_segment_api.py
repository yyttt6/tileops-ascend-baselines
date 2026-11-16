"""Original UnsortedSegmentSum public API; no replacement reduction/kernel."""
import ctypes as C,hashlib,json,os,math
from pathlib import Path
class SegmentNativeAPI:
 def __init__(self,torch,manifest,qualification=None):
  self.t=torch;self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-segment-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  p=self.root/'op_api/lib/libcust_opapi.so';assert hashlib.sha256(p.read_bytes()).hexdigest()==self.m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];self.lib=C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL);self.base=C.CDLL('libnnopbase.so')
  self.base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];self.base.aclCreateTensor.restype=C.c_void_p
  self.base.aclDestroyTensor.argtypes=[C.c_void_p];self.base.aclDestroyTensor.restype=C.c_int
  self.lib.aclnnUnsortedSegmentSumGetWorkspaceSize.argtypes=[C.c_void_p,C.c_void_p,C.c_int64,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];self.lib.aclnnUnsortedSegmentSumGetWorkspaceSize.restype=C.c_int
  self.lib.aclnnUnsortedSegmentSum.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.lib.aclnnUnsortedSegmentSum.restype=C.c_int
 def __call__(self,x,dims,output_shape,output_dtype):
  t=self.t;codes={t.float16:1,t.bfloat16:27,t.float32:0,t.int32:3,t.int64:9};assert x.device.type=='npu' and x.dtype in codes
  axes=tuple(d%x.ndim for d in dims);outer=tuple(d for d in range(x.ndim) if d not in axes);width=math.prod(x.shape[d] for d in axes);rows=x.numel()//width
  dtype=t.float32 if x.dtype in (t.float16,t.bfloat16,t.float32) else output_dtype
  xx=x.permute(*axes,*outer).contiguous().reshape(width,rows).to(dtype)
  ids=t.zeros(width,device=x.device,dtype=t.int32);num=t.tensor(1,device=x.device,dtype=t.int32);out=t.empty((1,rows),device=x.device,dtype=dtype)
  # Original binding constants={2:1} is a CPU int64 API attribute; retain
  # original per-call device num allocation above even though not consumed by C API.
  keep=[xx,ids,num,out];owned=[]
  def tensor(a):
   dims=(C.c_int64*a.ndim)(*a.shape);strides=(C.c_int64*a.ndim)(*a.stride());keep.extend([dims,strides]);p=self.base.aclCreateTensor(dims,a.ndim,codes[a.dtype],strides,0,2,dims,a.ndim,a.data_ptr());assert p;owned.append(p);return p
  try:
   a=tensor(xx);b=tensor(ids);y=tensor(out);size=C.c_uint64();exe=C.c_void_p();rc=self.lib.aclnnUnsortedSegmentSumGetWorkspaceSize(a,b,1,y,C.byref(size),C.byref(exe));assert rc==0,('UnsortedSegmentSum','GetWorkspaceSize',rc)
   workspace=t.empty(size.value,dtype=t.uint8,device=x.device) if size.value else None;keep.append(workspace);rc=self.lib.aclnnUnsortedSegmentSum(workspace.data_ptr() if workspace is not None else None,size.value,exe,t.npu.current_stream().npu_stream);assert rc==0,('UnsortedSegmentSum','run',rc);t.npu.synchronize();return out.reshape(tuple(output_shape)).to(output_dtype)
  finally:
   t.npu.synchronize()
   for p in reversed(owned):assert self.base.aclDestroyTensor(p)==0
