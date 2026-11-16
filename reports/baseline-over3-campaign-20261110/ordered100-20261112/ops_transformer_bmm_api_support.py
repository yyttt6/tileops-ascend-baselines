from pathlib import Path
import json,hashlib
from ops_transformer_bmm_api_graph_api import GraphCall
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-arithmetic-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  assert library=='ops-transformer' and case._call.signature.name=='BmmFwdOp'
  a,b=case.inputs;assert a.dtype in (torch.float16,torch.bfloat16) and b.dtype==a.dtype
  q=Path(__file__).with_name('TRANSFORMER_BMM_API_FORMAL_PREREQUISITES.json');assert json.loads(q.read_text())['original950_and_both_original_guarded_hostbuilds_verified']
  manifest=N/'TRANSFORMER_ARITHMETIC_PACKAGE.json';m=json.loads(manifest.read_text());vendor=Path(m['vendor']);sources={}
  for name,rec in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256'];sources[str(f)]=rec['sha256']
  binding=GraphCall('ops-transformer')
  def run(a, b):
          # Every slice, transpose and accumulator initialization belongs to run.
          outputs = []
          for lhs, rhs in zip(a, b, strict=True):
              group = torch.full((1,), lhs.shape[1], device=a.device, dtype=torch.int64)
              out = torch.zeros((1, lhs.shape[0], rhs.shape[1]), device=a.device, dtype=torch.float32)
              binding.call('GroupedMatmulAdd', [lhs.T.contiguous(), rhs.contiguous(), group, out], [out],
                           {'transpose_x': True, 'transpose_weight': False,
                            'group_type': 2, 'group_list_type': 0})
              outputs.append(out[0].to(a.dtype))
          return torch.stack(outputs)
  for f in [Path(__file__),Path(__file__).with_name('ops_transformer_bmm_api_graph_api.py'),q,manifest]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'ops-transformer::original GroupedMatmulAdd independent k-groups','OriginalBmm runASTexact; everytranspose/zeroFP32accumulator/groupmetadata/outputcast/stack timed; no newkernel/precision/math.',sources,cleanup=binding.cleanup)
 return build
