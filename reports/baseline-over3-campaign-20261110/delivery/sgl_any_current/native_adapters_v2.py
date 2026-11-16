import pathlib,sys,types,hashlib
ROOT=pathlib.Path(__file__).parent
def factory(op,params,fixed,args,*,case):
 import torch
 assert op=='AnyFwdOp' and not fixed and args[0].dtype==torch.bool
 sys.path.append('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
 pkg=types.ModuleType('sgl_kernel_npu');pkg.__path__=[str(ROOT/'sgl-source/sgl_kernel_npu')];sys.modules['sgl_kernel_npu']=pkg
 from sgl_kernel_npu.sample.argmax_softmax_prob import argmax_softmax_prob_fused
 from row_layout_original import row_layout
 pack,width,shape=row_layout(types.SimpleNamespace(inputs=args,params=case.params))
 def run(value):
  indices=argmax_softmax_prob_fused(pack(value.to(torch.float32)))[0].reshape(-1,1)
  return pack(value).gather(1,indices).reshape(shape)
 files=[ROOT/'native_adapters_v2.py',ROOT/'row_layout_original.py',*(ROOT/'source-review').glob('*.py'),*(ROOT/'sgl-source').rglob('*.py')]
 return run,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
