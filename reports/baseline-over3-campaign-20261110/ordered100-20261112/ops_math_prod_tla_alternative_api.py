from pathlib import Path
import ctypes as C,json,hashlib,inspect
from ops_transformer_gmmv5_api_support import _MathCall
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-prod-original-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  p=case.params
  assert library=='tilelang-ascend' and case._call.signature.name=='ProdFwdOp' and isinstance(p['dim'],int) and p['dim'] in [-1,0] and p['keepdim'] is False
  input_dtype=getattr(torch,p['x'][1]);output_dtype=getattr(torch,p['output'][1]);shape=tuple(p['output'][0])
  assert p['dtype'] is None and input_dtype==output_dtype or p['dtype']==p['output'][1]
  api='aclnnProdDim';vendor=N/'private-v1/vendors/originalprod_math';m=json.loads((N/'PROD_ORIGINAL_PACKAGE.json').read_text());sources={}
  for name,r in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];sources[str(f)]=r['sha256']
  for f in [Path(__file__),Path(inspect.getfile(_MathCall)),N/'PROD_ORIGINAL_PACKAGE.json']:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  b=_MathCall(api,[C.c_void_p,C.c_int64,C.c_bool,C.c_int,C.c_void_p],library='ops-math',host_library=vendor/'op_api/lib/libcust_opapi.so')
  def run(x):
   assert x.dtype==input_dtype and tuple(x.shape)==tuple(p['x'][0]);record=b.record();out=torch.empty(shape,dtype=output_dtype,device=x.device)
   try:
    b.call([b.tensor(x,record),p['dim'],p['keepdim'],b.codes[output_dtype],b.tensor(out,record)],record,x.device);return out
   finally:b.cleanup()
  return Candidate(Implementation(run),'ops-math::'+api,'Original complete ops-math ProdDim API for unchanged TLA Prod contract; original dim/keepdim and explicit output dtype passed to native API. No outside reduction or dtype conversion; original NPU checker.',sources,timing_mode='host_roundtrip',cleanup=b.cleanup)
 return build
