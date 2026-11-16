import pathlib,sys,types,hashlib
ROOT=pathlib.Path(__file__).parent
def factory(op,params,fixed,args,*,case):
 assert op=='HardsigmoidFwdOp' and not fixed
 sys.path.append('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
 pkg=types.ModuleType('sgl_kernel_npu');pkg.__path__=[str(ROOT/'sgl-source/sgl_kernel_npu')];sys.modules['sgl_kernel_npu']=pkg
 from sgl_kernel_npu.sample.argmax_softmax_prob import argmax_softmax_prob_fused
 from sgl_kernel_npu.moe.mul_add import mul_add
 from original_hardsigmoid import build
 run=build(argmax_softmax_prob_fused,mul_add)
 files=[ROOT/'native_adapters_v2.py',ROOT/'original_hardsigmoid.py',ROOT/'original_sub_body.py',*(ROOT/'source-review').glob('*.py'),*(ROOT/'sgl-source').rglob('*.py')]
 return run,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
