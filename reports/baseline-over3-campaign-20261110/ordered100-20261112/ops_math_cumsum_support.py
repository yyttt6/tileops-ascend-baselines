"""Full original same-pin Cumsum API for exact frozen case parameters."""
from pathlib import Path
import os,json,hashlib
P=Path(__file__).resolve().parent;N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-cumscan-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from ops_math_cumsum_api import CumsumAPI
  assert library=='ops-math' and case._call.signature.name=='CumsumFwdOp';p=case.params;assert not p.get('exclusive',False) and not p.get('reverse',False);dim=p.get('dim',-1);assert isinstance(dim,int);assert tuple(p['x'][0])==tuple(p['y'][0]) and p['x'][1]==p['y'][1] and p['x'][1] in ('float16','bfloat16'), 'Exact frozen x/y same-shape/dtype contract only'
  f=N/'CUMSUM_HOST_QUALIFICATION.json';q=json.loads(f.read_text());assert q['full_API_and_original_flush_and_private_kernel_verified'] and q['execution_hostname']==os.uname().nodename;mf=N/'MATH_CUMSCAN_PACKAGE.json';assert hashlib.sha256(mf.read_bytes()).hexdigest()==q['native_package_manifest_sha256']
  for name,h in q['qualified_API_source_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
  api=CumsumAPI(torch,dim);sources={**api.sources,str(f):hashlib.sha256(f.read_bytes()).hexdigest()}
  for name in ['ops_math_cumsum_api.py','ops_math_cumsum_support.py','ops_math_l1_api.py']:
   f=P/name;sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(api),'ops-math complete aclnnCumsum, original axis/input/output dtype','Complete contiguous/descriptor/workspace/output/nativeAPI timed; no Torch scan, kernel or checker changes.',sources,cleanup=api.cleanup)
 return build
