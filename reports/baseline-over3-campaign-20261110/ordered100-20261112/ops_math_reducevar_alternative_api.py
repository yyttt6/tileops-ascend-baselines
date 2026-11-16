from pathlib import Path
import ctypes as C,json,hashlib
from ops_transformer_gmmv5_api_support import _MathCall
N=Path("/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-reducevar-original-v1")
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  assert library=='ops-nn' and case._call.signature.name in ['VarFwdOp','VarMeanFwdOp'] and case.id=='mlp-intermediate-float32';p=case.params;assert p['correction']==1 and p['dim']==-1 and not p['keepdim'];both=case._call.signature.name=='VarMeanFwdOp';api='aclnnVarMean' if both else 'aclnnVarCorrection';vendor=N/'private-v1/vendors/originalvar_math';m=json.loads((N/'REDUCEVAR_ORIGINAL_PACKAGE.json').read_text());sources={}
  for name,r in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];sources[str(f)]=r['sha256']
  for f in [Path(__file__),N/'REDUCEVAR_ORIGINAL_PACKAGE.json'] :sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  b=_MathCall(api,[C.c_void_p,C.c_void_p,C.c_int64,C.c_bool]+[C.c_void_p]*(2 if both else 1),library='ops-math',host_library=vendor/'op_api/lib/libcust_opapi.so')
  def run(x):
   assert x.dtype==torch.float32 and tuple(x.shape)==(2048,11008);record=b.record();var=torch.empty((2048,),dtype=x.dtype,device=x.device);mean=torch.empty_like(var) if both else None
   try:
    args=[b.tensor(x,record),b.array([-1],record),1,False,b.tensor(var,record)]
    if both:args.append(b.tensor(mean,record))
    b.call(args,record,x.device);return (var,mean) if both else var
   finally:b.cleanup()
  return Candidate(Implementation(run),'ops-math::'+api,'Alternative complete original ops-math API for original NN contract; exact correction1/FP32/outputs; not retired LayerNorm composition or original NN implementation.',sources,timing_mode='host_roundtrip',cleanup=b.cleanup)
 return build
