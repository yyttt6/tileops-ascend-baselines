from pathlib import Path
import ctypes as C,json,hashlib,inspect
from ops_transformer_gmmv5_api_support import _MathCall
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-argextrema-original-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  p=case.params;op=case._call.signature.name
  assert library=='tilelang-ascend' and op in ['ArgmaxFwdOp','ArgminFwdOp'] and p['dim']==-1 and p['keepdim'] is False and p['output'][1]=='int64'
  dtype=getattr(torch,p['x'][1]);shape=tuple(p['output'][0]);api={'ArgmaxFwdOp':'aclnnArgMax','ArgminFwdOp':'aclnnArgMin'}[op]
  vendor=N/'private-v1/vendors/originalargextrema_math';m=json.loads((N/'ARGEXTREMA_ORIGINAL_PACKAGE.json').read_text());sources={}
  for name,r in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];sources[str(f)]=r['sha256']
  for f in [Path(__file__),Path(inspect.getfile(_MathCall)),N/'ARGEXTREMA_ORIGINAL_PACKAGE.json']:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  b=_MathCall(api,[C.c_void_p,C.c_int64,C.c_bool,C.c_void_p],library='ops-math',host_library=vendor/'op_api/lib/libcust_opapi.so')
  def run(x):
   assert x.dtype==dtype and tuple(x.shape)==tuple(p['x'][0]);record=b.record();out=torch.empty(shape,dtype=torch.int64,device=x.device)
   try:
    b.call([b.tensor(x,record),p['dim'],p['keepdim'],b.tensor(out,record)],record,x.device);return out
   finally:b.cleanup()
  return Candidate(Implementation(run),'ops-math::'+api,'Original complete ops-math ArgMax/ArgMin API for unchanged TLA contract; native INT64 output and original NPU index/tie checker. No outside cast, negate, reduction or tie repair.',sources,timing_mode='host_roundtrip',cleanup=b.cleanup)
 return build
