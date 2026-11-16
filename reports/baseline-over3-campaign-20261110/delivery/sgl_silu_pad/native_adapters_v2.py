import pathlib,sys,types,hashlib
ROOT=pathlib.Path(__file__).parent
def factory(op,params,fixed,args,*,case):
 assert op=='SiluAndMulFwdOp'
 sys.path.append('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
 pkg=types.ModuleType('sgl_kernel_npu');pkg.__path__=[str(ROOT/'sgl-source/sgl_kernel_npu')];sys.modules['sgl_kernel_npu']=pkg
 from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant
 if fixed:
  from candidate_silu import build
 else:
  from original_silu import build
 run=build(types.SimpleNamespace(inputs=args,params=case.params),swiglu_quant)
 files=[ROOT/'native_adapters_v2.py',ROOT/'original_silu.py',ROOT/'candidate_silu.py',*(ROOT/'source-review').glob('*.py'),*(ROOT/'sgl-source').rglob('*.py')]
 return run,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
