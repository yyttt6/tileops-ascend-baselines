import pathlib,sys,types,hashlib
ROOT=pathlib.Path(__file__).parent
def factory(op,params,fixed,args,*,case):
 assert op=='INT8DequantPerTensorFwdOp'
 import torch
 sys.path.append('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
 pkg=types.ModuleType('sgl_kernel_npu');pkg.__path__=[str(ROOT/'sgl-source/sgl_kernel_npu')];sys.modules['sgl_kernel_npu']=pkg
 from sgl_kernel_npu.norm.scale_shift import fused_scale_shift
 from original_dequant import build
 run=build(fused_scale_shift,op,getattr(torch,case.params['out_dtype']))
 if fixed:
  from natural_scale_candidate import build as build_fixed
  run=build_fixed(fused_scale_shift,getattr(torch,case.params['out_dtype']))
 files=[ROOT/'native_adapters_v2.py',ROOT/'original_dequant.py',ROOT/'natural_scale_candidate.py',*(ROOT/'source-review').glob('*.py'),*(ROOT/'sgl-source').rglob('*.py')]
 return run,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
