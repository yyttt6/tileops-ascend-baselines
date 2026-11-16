from pathlib import Path
import hashlib,json
from ops_math_leaf4_api import LeafAPI,N
MAP={'BitwiseAndFwdOp':'LogicalAnd'}
def candidate_factory(case,library):
 def build():
  assert library=='ops-math';op=case._call.signature.name;assert op in MAP
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  q=N/'LEAF4_TWO_HOST_QUALIFICATION.json';assert json.loads(q.read_text())['two_host_functional_and_private_kernel_verified'] is True
  expected=torch.bool if op=='LogicalAndFwdOp' else case.inputs[0].dtype;assert getattr(torch,case.params['output'][1])==expected
  assert all(x.dtype==torch.bool for x in case.inputs),'BOOL-only exact bitwise/logical equivalence';api=LeafAPI(torch,MAP[op]);sources={}
  for f in [Path(__file__),Path(__file__).with_name('ops_math_leaf4_api.py'),N/'LEAF4_PACKAGE.json',q]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  for f,r in api.m['files'].items():sources[str(N/'private-v1'/f)]=r['sha256']
  return Candidate(Implementation(api),'ops-math::full original aclnn'+MAP[op]+'GetWorkspaceSize+aclnn'+MAP[op],'Every original native internal cast/contiguous/ViewCopy and all per-call descriptors, output/workspace allocation, broadcast and cleanup retained. No cached values or Torch arithmetic substitute.',sources,cleanup=api.cleanup)
 return build
