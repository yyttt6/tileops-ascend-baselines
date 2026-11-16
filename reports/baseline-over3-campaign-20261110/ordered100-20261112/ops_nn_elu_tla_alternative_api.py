from pathlib import Path
import ctypes as C,json,hashlib,inspect
from ops_transformer_gmmv5_api_support import _MathCall
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-elu-original-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  p=case.params
  assert library=='tilelang-ascend' and case._call.signature.name=='EluFwdOp' and p['inplace'] is False and p['alpha']==1.0
  assert p['input']==p['output'];shape=tuple(p['input'][0]);dtype=getattr(torch,p['input'][1]);api='aclnnElu'
  vendor=N/'private-v1/vendors/originalelu_nn';m=json.loads((N/'ELU_ORIGINAL_PACKAGE.json').read_text());sources={}
  for name,r in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];sources[str(f)]=r['sha256']
  for f in [Path(__file__),Path(inspect.getfile(_MathCall)),N/'ELU_ORIGINAL_PACKAGE.json']:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  b=_MathCall(api,[C.c_void_p]*5,library='ops-nn',host_library=vendor/'op_api/lib/libcust_opapi.so')
  b.base.aclCreateScalar.argtypes=[C.c_void_p,C.c_int];b.base.aclCreateScalar.restype=C.c_void_p
  b.base.aclDestroyScalar.argtypes=[C.c_void_p];b.base.aclDestroyScalar.restype=C.c_int
  def run(x):
   assert x.dtype==dtype and tuple(x.shape)==shape;record=b.record();out=torch.empty(shape,dtype=x.dtype,device=x.device);scalars=[]
   try:
    for value in [p['alpha'],1.0,1.0]:
     storage=C.c_float(value);record['keep'].append(storage);ptr=b.base.aclCreateScalar(C.byref(storage),0)
     if not ptr:raise RuntimeError('aclCreateScalar returned null')
     scalars.append(ptr)
    b.call([b.tensor(x,record),*scalars,b.tensor(out,record)],record,x.device);return out
   finally:
    try:b.cleanup()
    finally:
     for ptr in scalars:b.check(b.base.aclDestroyScalar(ptr),'aclDestroyScalar')
  return Candidate(Implementation(run),'ops-nn::'+api,'Original complete ops-nn aclnnElu API for unchanged TLA ELU contract; original alpha and native scale/inputScale defaults. Full native math and original NPU checker; explicitly alternative implementation.',sources,timing_mode='host_roundtrip',cleanup=b.cleanup)
 return build
