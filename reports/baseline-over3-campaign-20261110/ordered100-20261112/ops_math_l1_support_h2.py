from pathlib import Path
import hashlib,json
from ops_math_l1_api import MathL1API,N
def candidate_factory(case,library):
 def build():
  assert library=='ops-math' and case._call.signature.name=='VectorNormFwdOp' and case.params['ord']==1 and case.params['keepdim'] is False
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  q=N/'MATHL1_H2_EXECUTION_QUALIFICATION.json';assert json.loads(q.read_text())['host2_functional_and_private_kernel_verified'] is True
  dtype=getattr(torch,case.params['output'][1]);assert dtype==case.inputs[0].dtype or case.params['dtype']=='float32' and dtype==torch.float32
  api=MathL1API(torch,dim=case.params['dim'],shape=case.params['output'][0],outdtype=dtype);sources=dict(api.sources)
  for f in [Path(__file__),Path(__file__).with_name('ops_math_l1_api.py'),q]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(api),'ops-math::full original aclnnAbs + full original aclnnReduceSum','Two same-pin/sameSDK original complete publicAPIs, full nativecasts/contiguous/ViewCopy/axispacking/descriptors/workspaces/output. Sum dtype follows original declared VectorNormdtype; no inputprecision promotion or torchabs/sum. Original kernels unmodified.',sources,cleanup=api.cleanup)
 return build
