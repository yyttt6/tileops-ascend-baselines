import pathlib,sys,types,hashlib,math
ROOT=pathlib.Path(__file__).parent
def factory(op,params,fixed,args,*,case):
 assert op=='GroupNormFwdOp' and not fixed
 import torch
 sys.path.append('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
 pkg=types.ModuleType('sgl_kernel_npu');pkg.__path__=[str(ROOT/'sgl-source/sgl_kernel_npu')];sys.modules['sgl_kernel_npu']=pkg
 from sgl_kernel_npu.fla.layernorm_gated import layer_norm_fwd_npu_simd,layer_norm_fwd_npu
 from original_groupnorm import build
 eps=case.params.get('eps');eps=torch.finfo(torch.float32).eps if eps is None else eps
 from types import SimpleNamespace
 run=build(layer_norm_fwd_npu_simd,SimpleNamespace(inputs=args,params=case.params))
 files=[ROOT/'native_adapters_v2.py',ROOT/'original_groupnorm.py',*(ROOT/'source-review').glob('*.py'),*(ROOT/'sgl-source').rglob('*.py')]
 return run,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
