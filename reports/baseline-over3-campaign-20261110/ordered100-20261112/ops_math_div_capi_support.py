"""Complete original native Div API, fresh call each timed invocation."""
from pathlib import Path
import json,hashlib
from ops_math_div_capi import DivAPI
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-div-v1')
def candidate_factory(case,library):
 def build():
  assert library=='ops-math' and case._call.signature.name=='DivFwdOp' and case.params.get('rounding_mode') is None
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  proof=json.loads((N/'DIV_TWO_HOST_QUALIFICATION.json').read_text());assert proof['two_host_functional_and_private_kernel_verified'] is True
  api=DivAPI(torch);source={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  for f in [Path(__file__).with_name('ops_math_div_capi.py'),N/'DIV_PACKAGE.json',N/'DIV_TWO_HOST_QUALIFICATION.json']:
   source[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  for f,r in api.m['files'].items():source[str(N/'graph-private-v1'/f)]=r['sha256']
  for f in (N/'source/math/div').rglob('*'):
   if f.is_file():source[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(api),'ops-math::original aclnnDivGetWorkspaceSize+aclnnDiv','Complete original two-stage API, all per-call contiguous inputs, full output/workspace allocations, descriptors and native call. Equal dtype, original broadcast; no Torch division.',source,cleanup=api.cleanup)
 return build
