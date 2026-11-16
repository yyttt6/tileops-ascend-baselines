"""Original complete Cumprod CAPI and explicitly declared SDK AICPU dependency."""
from pathlib import Path
import os,json,hashlib
from ops_math_cumprod_api import ScanAPI,N
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  assert library=='ops-math' and case._call.signature.name=='CumprodFwdOp'
  p=case.params;assert p['x']==p['y'] and p['dim']==-1
  q=Path(__file__).with_name('CUMPROD_FORMAL_PREREQUISITES.json');proof=json.loads(q.read_text());assert proof['complete_API_short3dtype_guard_valid'] and proof['same_original_private_API_bothhosts'];assert proof['scope']==[252,253,254,255]
  expected=getattr(torch,p['y'][1]);api=ScanAPI(torch,'Cumprod',dim=p['dim'],outdtype=expected);sources={}
  for f in [Path(__file__),Path(__file__).with_name('ops_math_cumprod_api.py'),N/'SCANBOOL_PACKAGE_V2.json',q,Path(proof['AICPU_config_path'])]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  assert sources[proof['AICPU_config_path']]==proof['AICPU_config_sha256']
  for f,r in api.m['files'].items():sources[str(N/'private-v2'/f)]=r['sha256']
  return Candidate(Implementation(api),'ops-math::original complete aclnnCumprod declared SDK TFKernel/AICPU','Original contiguous/cast/ViewCopy/descriptor/workspace/output complete; isolated full32 retains AICPU/copy. No privateAIcore claim or Torchscan substitute.',sources,cleanup=api.cleanup)
 return build
