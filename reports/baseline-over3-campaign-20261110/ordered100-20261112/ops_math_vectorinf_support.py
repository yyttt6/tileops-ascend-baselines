from pathlib import Path
import hashlib,json
from ops_math_vectorinf_api import VectorInfAPI,N
def candidate_factory(case,library):
 def build():
  assert library=='ops-math' and case._call.signature.name=='VectorNormFwdOp' and case.params['ord']==float('inf') and case.params['keepdim'] is False
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  q=N/'VECTORINF_TWO_HOST_QUALIFICATION.json';assert json.loads(q.read_text())['two_host_functional_and_private_kernel_verified'] is True
  dtype=getattr(torch,case.params['output'][1]);assert dtype==case.inputs[0].dtype or case.params['dtype']=='float32' and dtype==torch.float32
  api=VectorInfAPI(torch,dim=case.params['dim'],shape=case.params['output'][0],outdtype=dtype);sources={}
  for f in [Path(__file__),Path(__file__).with_name('ops_math_vectorinf_api.py'),N/'VECTORINF_PACKAGE.json',q]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  for f,r in api.m['files'].items():sources[str(N/'private-v1'/f)]=r['sha256']
  return Candidate(Implementation(api),'ops-math::full original aclnnAbs + full original aclnnAmax','All original contiguous/ViewCopy/axes, per-call descriptors/output/workspace and declared finaloutputcast timed. Abs/max of BF16values commute with exactFP32embedding; requestedFP32output honored without changing input/checkerprecision. No torch abs/max or newkernel.',sources,cleanup=api.cleanup)
 return build
