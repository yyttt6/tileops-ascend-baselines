from pathlib import Path
import hashlib,json
from ops_math_countnz_api import CountNZAPI,N
def candidate_factory(case,library):
 def build():
  assert library=='ops-math' and case._call.signature.name=='CountNonzeroFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  q=N/'COUNTNZ_TWO_HOST_QUALIFICATION.json';assert json.loads(q.read_text())['two_host_functional_and_private_kernel_verified'] is True
  assert case.params['dim']==-1 and case.params['output'][1]=='int64' and len(case.inputs)==1 and case.inputs[0].ndim==2
  assert case.inputs[0].shape[-1] in (32768,4096,11008)
  api=CountNZAPI(torch,dim=case.params['dim'],shape=case.params['output'][0]);sources={}
  for f in [Path(__file__),Path(__file__).with_name('ops_math_countnz_api.py'),N/'COUNTNZ_PACKAGE.json',q]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  for f,r in api.m['files'].items():sources[str(N/'private-v1'/f)]=r['sha256']
  return Candidate(Implementation(api),'ops-math::full original aclnnNeTensor + full original aclnnReduceSum','Per-call zero, descriptors/workspaces/output, original native casts/contiguous/ViewCopy and INT64output retained. Original BOOLsum FP32 exact0/1 branch below2^24 unchanged; no Torch predicate/reduction or kernel change.',sources,cleanup=api.cleanup)
 return build
