import pathlib,sys,types,hashlib
ROOT=pathlib.Path(__file__).parent
def factory(op,params,fixed,args,*,case):
 import torch
 assert op=='AmaxFwdOp' and not fixed
 sys.path.append('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
 pkg=types.ModuleType('sgl_kernel_npu');pkg.__path__=[str(ROOT/'sgl-source/sgl_kernel_npu')];sys.modules['sgl_kernel_npu']=pkg
 from sgl_kernel_npu.sample.argmax_softmax_prob import argmax_softmax_prob_fused
 from original_amax import row_max_case,native_factory
 proxy=types.SimpleNamespace(inputs=args,params=case.params)
 run=row_max_case(proxy,lambda metadata:native_factory(metadata,argmax_softmax_prob_fused))
 files=[ROOT/'native_adapters_v2.py',ROOT/'original_amax.py',*(ROOT/'source-review').glob('*.py'),*(ROOT/'sgl-source').rglob('*.py')]
 return run,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
