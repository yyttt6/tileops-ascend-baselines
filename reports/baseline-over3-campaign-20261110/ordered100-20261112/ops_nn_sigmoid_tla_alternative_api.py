from pathlib import Path
import ctypes as C,json,hashlib,inspect
from ops_transformer_gmmv5_api_support import _MathCall
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-sigmoid-original-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  p=case.params
  assert library=='tilelang-ascend' and case._call.signature.name=='SigmoidFwdOp'
  assert p['input']==p['output'] and p['input'][1] in ['float16','bfloat16'];shape=tuple(p['input'][0]);dtype=getattr(torch,p['input'][1]);api='aclnnSigmoid'
  vendor=N/'private-v1/vendors/originalsigmoid_nn';m=json.loads((N/'SIGMOID_ORIGINAL_PACKAGE.json').read_text());sources={}
  for name,r in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];sources[str(f)]=r['sha256']
  for f in [Path(__file__),Path(inspect.getfile(_MathCall)),N/'SIGMOID_ORIGINAL_PACKAGE.json']:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  b=_MathCall(api,[C.c_void_p,C.c_void_p],library='ops-nn',host_library=vendor/'op_api/lib/libcust_opapi.so')
  def run(x):
   assert x.dtype==dtype and tuple(x.shape)==shape;record=b.record();out=torch.empty(shape,dtype=x.dtype,device=x.device)
   try:
    b.call([b.tensor(x,record),b.tensor(out,record)],record,x.device);return out
   finally:b.cleanup()
  return Candidate(Implementation(run),'ops-nn::'+api,'Original complete ops-nn Sigmoid API for unchanged TLA contract; original storage dtype, no external cast/math. Original NPU reference and full checker retained; explicitly alternative implementation.',sources,timing_mode='host_roundtrip',cleanup=b.cleanup)
 return build
