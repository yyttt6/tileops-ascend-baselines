from pathlib import Path
import ctypes as C,json,hashlib
from ops_transformer_gmmv5_api_support import _MathCall
N=Path("/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-logsumexp-original-v1")
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  assert library=='tilelang-ascend' and case._call.signature.name=='LogSumExpFwdOp';p=case.params;assert p['dim']==-1 and not p['keepdim'];dims=[p['dim']] if isinstance(p['dim'],int) else list(p['dim']);shape=tuple(p['output'][0]);dtype=getattr(torch,p['x'][1]);both=False;api='aclnnLogSumExp';vendor=N/'private-v1/vendors/originallogsumexp_math';m=json.loads((N/'LOGSUMEXP_ORIGINAL_PACKAGE.json').read_text());sources={}
  for name,r in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];sources[str(f)]=r['sha256']
  for f in [Path(__file__),Path(__import__('inspect').getfile(_MathCall)),N/'LOGSUMEXP_ORIGINAL_PACKAGE.json'] :sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  b=_MathCall(api,[C.c_void_p,C.c_void_p,C.c_bool]+[C.c_void_p]*(2 if both else 1),library='ops-math',host_library=vendor/'op_api/lib/libcust_opapi.so')
  def run(x):
   assert x.dtype==dtype and tuple(x.shape)==tuple(p['x'][0]);record=b.record();var=torch.empty(shape,dtype=x.dtype,device=x.device);mean=torch.empty_like(var) if both else None
   try:
    args=[b.tensor(x,record),b.array(dims,record),p['keepdim'],b.tensor(var,record)]
    if both:args.append(b.tensor(mean,record))
    b.call(args,record,x.device);return (var,mean) if both else var
   finally:b.cleanup()
  return Candidate(Implementation(run),'ops-math::'+api,'Alternative complete original ops-math API for original TLA LogSumExp contract; exact fullaxis/shapes/dtypes/outputs; not retired LayerNorm composition or original TLA implementation.',sources,timing_mode='host_roundtrip',cleanup=b.cleanup)
 return build
