from pathlib import Path
import ctypes as C,json,hashlib,inspect
from ops_transformer_gmmv5_api_support import _MathCall
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-mean-original-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  p=case.params
  assert library=='tilelang-ascend' and case._call.signature.name=='MeanFwdOp' and p['dim']==-1 and p['keepdim'] is False and p['dtype'] is None
  assert p['x'][1]==p['output'][1]=='bfloat16';shape=tuple(p['output'][0]);api='aclnnMean'
  vendor=N/'private-v1/vendors/allpairs_math_poolmean_math';m=json.loads((N/'MEAN_ORIGINAL_PACKAGE.json').read_text());sources={}
  for name,r in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];sources[str(f)]=r['sha256']
  for f in [Path(__file__),Path(inspect.getfile(_MathCall)),N/'MEAN_ORIGINAL_PACKAGE.json']:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  b=_MathCall(api,[C.c_void_p,C.c_void_p,C.c_bool,C.c_int,C.c_void_p],library='ops-math',host_library=vendor/'op_api/lib/libcust_opapi.so')
  def run(x):
   assert x.dtype==torch.bfloat16 and tuple(x.shape)==tuple(p['x'][0]);record=b.record();out=torch.empty(shape,dtype=x.dtype,device=x.device)
   try:
    b.call([b.tensor(x,record),b.array([p['dim']],record),p['keepdim'],b.codes[out.dtype],b.tensor(out,record)],record,x.device);return out
   finally:b.cleanup()
  return Candidate(Implementation(run),'ops-math::'+api,'Original complete ops-math Mean API for unchanged TLA BF16 contract; original dim/keepdim and BF16 input/output, no external FP32 conversion or tensor arithmetic. Original NPU checker retained.',sources,timing_mode='host_roundtrip',cleanup=b.cleanup)
 return build
