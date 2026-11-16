import pathlib,sys,types,hashlib
ROOT=pathlib.Path(__file__).parent
def factory(op,params,fixed,args,*,case):
 import torch
 assert op in ('AnyFwdOp','AllFwdOp') and not fixed and args[0].dtype in (torch.float16,torch.bfloat16,torch.float32)
 sys.path.append('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
 pkg=types.ModuleType('sgl_kernel_npu');pkg.__path__=[str(ROOT/'sgl-source/sgl_kernel_npu')];sys.modules['sgl_kernel_npu']=pkg
 from sgl_kernel_npu.sample.argmax_softmax_prob import argmax_softmax_prob_fused
 from sgl_kernel_npu.moe.mul_add import mul_add
 from finite_predicate import build as build_ne
 from original_sub_body import build_sub
 from row_layout_original import row_layout
 pack,width,shape=row_layout(types.SimpleNamespace(inputs=args,params=case.params))
 ne=build_ne(argmax_softmax_prob_fused,mul_add)
 if op=='AllFwdOp':
  flag=torch.empty(args[0].shape,device=args[0].device,dtype=torch.bool)
  sub=build_sub(types.SimpleNamespace(inputs=(torch.zeros_like(flag),flag),params={'alpha':1}),mul_add)
 def negative(x):
  return sub(torch.zeros_like(x),x)
 def run(value):
  flags=ne(value)
  x=flags.to(torch.float32)
  if op=='AllFwdOp':x=negative(x)
  indices=argmax_softmax_prob_fused(pack(x))[0].reshape(-1,1)
  return pack(flags).gather(1,indices).reshape(shape)
 files=[ROOT/'native_adapters_v2.py',ROOT/'row_layout_original.py',ROOT/'original_sub_body.py',ROOT/'finite_predicate.py',*(ROOT/'source-review').glob('*.py'),*(ROOT/'sgl-source').rglob('*.py')]
 return run,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
