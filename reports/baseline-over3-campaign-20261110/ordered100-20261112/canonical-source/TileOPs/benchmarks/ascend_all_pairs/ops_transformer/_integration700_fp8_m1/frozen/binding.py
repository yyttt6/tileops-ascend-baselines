"""M=1 FP8 block-scale GEMM by operand exchange and output transpose only."""
import ctypes as C
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent

def build(case):
 import torch
 from benchmarks.ascend_math_more import _MathCall
 from benchmarks.ascend_baselines import Candidate,source_hashes
 from benchmarks.api import Implementation
 assert case.case_id == 'ds-v3-m1-down-blk-float8_e4m3fn-bfloat16' if hasattr(case,'case_id') else True
 b=_MathCall('aclnnGroupedMatmulV5',[C.c_void_p]*12+[C.c_int64]*4+[C.c_void_p]*4,library='ops-transformer',host_library=P/'vendor/op_api/lib/libcust_opapi.so')
 base=b.base;base.aclCreateTensorList.argtypes=[C.POINTER(C.c_void_p),C.c_uint64];base.aclCreateTensorList.restype=C.c_void_p;base.aclDestroyTensorList.argtypes=[C.c_void_p];base.aclDestroyTensorList.restype=C.c_int
 lists=[]; calls=[]
 def tl(ts,r):
  ptrs=[b.tensor(t,r) for t in ts];arr=(C.c_void_p*len(ptrs))(*ptrs);r['keep'].append(arr);p=base.aclCreateTensorList(arr,len(ptrs));assert p
  for q in ptrs:r['tensors'].remove(q)
  lists.append(p);return p
 def run(a,w,sa,sb):
  assert a.shape[0]==1 and sa.shape==(1,a.shape[1]//128)
  lhs=w.contiguous();rhs=a.T.contiguous().unsqueeze(0);scale_b=sa.T.contiguous().unsqueeze(0);scale_a=sb.contiguous()
  out=torch.empty((w.shape[0],1),dtype=torch.bfloat16,device=a.device);groups=torch.tensor([w.shape[0]],dtype=torch.int64,device=a.device)
  r=b.record();args=[tl([lhs],r),tl([rhs],r),None,tl([scale_b],r),None,None,None,tl([scale_a],r),b.tensor(groups,r),None,None,None,3,0,0,0,None,tl([out],r),None,None]
  evidence={'api':b.api,'workspace_pointer':C.cast(b.size,C.c_void_p).value,'launch_pointer':C.cast(b.launch,C.c_void_p).value,'binary':str(b.binary),'inode':b.binary.stat().st_ino,'binary_sha256':hashlib.sha256(b.binary.read_bytes()).hexdigest(),'inputs':[list(t.shape) for t in (a,w,sa,sb)],'native_shapes':[list(t.shape) for t in (lhs,rhs,scale_a,scale_b,out)],'maps':Path('/proc/self/maps').read_text(),'launched':False}
  calls.append(evidence);(P/'native.json').write_text(json.dumps(calls,indent=2))
  b.call(args,r,a.device);torch.npu.synchronize();evidence['launched']=True;evidence['maps_after']=Path('/proc/self/maps').read_text();(P/'native.json').write_text(json.dumps(calls,indent=2))
  return out.T.contiguous()
 def cleanup():
  torch.npu.synchronize()
  while lists:b.check(base.aclDestroyTensorList(lists.pop()),'aclDestroyTensorList')
  b.cleanup()
 files=[Path(__file__)]+[p for p in (P/'vendor').rglob('*') if p.is_file()]
 return Candidate(Implementation(run),'ops-transformer::aclnnGroupedMatmulV5','Exact M=1 operand swap, scale transpose and output transpose; no arithmetic outside complete original API.',source_hashes(*files),cleanup=cleanup)
