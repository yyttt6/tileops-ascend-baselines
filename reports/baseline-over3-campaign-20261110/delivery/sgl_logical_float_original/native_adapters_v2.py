import pathlib,sys,types,hashlib
ROOT=pathlib.Path(__file__).parent
def factory(op,params,fixed,args,*,case):
 assert op=='LogicalAndFwdOp'
 import torch
 assert all(x.dtype in (torch.float16,torch.bfloat16,torch.float32) for x in args)
 sys.path.append('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
 pkg=types.ModuleType('sgl_kernel_npu');pkg.__path__=[str(ROOT/'sgl-source/sgl_kernel_npu')];sys.modules['sgl_kernel_npu']=pkg
 from sgl_kernel_npu.norm.scale_shift import fused_scale_shift
 from sgl_kernel_npu.sample.argmax_softmax_prob import argmax_softmax_prob_fused
 from sgl_kernel_npu.moe.mul_add import mul_add
 from finite_predicate import build
 ne=build(argmax_softmax_prob_fused,mul_add)
 def run(x,y):
  x=ne(x).float();y=ne(y).float()
  if fixed:
   assert x.ndim==4 and y.ndim==3 and tuple(y.shape)==(x.shape[1],1,1)
   n,c,h,w=x.shape
   xx=x.contiguous().reshape(n,1,c*h*w)
   yy=y.expand(c,h,w).contiguous().reshape(-1)
   result=fused_scale_shift(xx,yy,torch.zeros_like(xx),scale_constant=0.0).reshape(x.shape)
  else:
   shape=torch.broadcast_shapes(x.shape,y.shape)
   xx=x.expand(shape).contiguous().reshape(1,1,-1)
   yy=y.expand(shape).contiguous().reshape(-1)
   result=fused_scale_shift(xx,yy,torch.zeros_like(xx),scale_constant=0.0).reshape(shape)
  return result.to(torch.bool)
 files=[ROOT/'native_adapters_v2.py',ROOT/'finite_predicate.py',*(ROOT/'source-review').glob('*.py'),*(ROOT/'sgl-source').rglob('*.py')]
 return run,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
