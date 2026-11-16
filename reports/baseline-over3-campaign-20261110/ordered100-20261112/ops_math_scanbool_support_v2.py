from pathlib import Path
import hashlib,json
from ops_math_scanbool_api_v2 import ScanAPI,N
MAP={'CumsumFwdOp':'Cumsum','CumprodFwdOp':'Cumprod','BitwiseOrFwdOp':'LogicalOr','BitwiseXorFwdOp':'NeTensor'}
def candidate_factory(case,library):
 def build():
  assert library=='ops-math';op=case._call.signature.name;assert op in MAP
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  q=N/'SCANBOOL_V2_TWO_HOST_QUALIFICATION.json';assert json.loads(q.read_text())['two_host_functional_and_private_kernel_verified'] is True
  expected=getattr(torch,case.params['output'][1]);assert expected in (torch.float16,torch.bfloat16,torch.float32,torch.bool)
  if op.startswith('Bitwise'):assert all(x.dtype==torch.bool for x in case.inputs) and expected==torch.bool
  api=ScanAPI(torch,MAP[op],dim=case.params.get('dim'),outdtype=expected);sources={}
  for f in [Path(__file__),Path(__file__).with_name('ops_math_scanbool_api_v2.py'),N/'SCANBOOL_PACKAGE_V2.json',q]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  for f,r in api.m['files'].items():sources[str(N/'private-v2'/f)]=r['sha256']
  return Candidate(Implementation(api),'ops-math::full original aclnn'+MAP[op]+'GetWorkspaceSize+aclnn'+MAP[op],'Every original native internal cast/contiguous/ViewCopy and all per-call descriptors, output/workspace allocation, broadcast and cleanup retained. No cached values or Torch arithmetic substitute.',sources,cleanup=api.cleanup)
 return build
