from pathlib import Path
import json,hashlib
from ops_transformer_rms_api_graph_api import GraphCall
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-arithmetic-v1')
def original_native(case):
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 q=Path(__file__).with_name('TRANSFORMER_RMS_API_FORMAL_PREREQUISITES.json');assert json.loads(q.read_text())['original950_and_both_original_guarded_hostbuilds_verified']
 manifest=N/'TRANSFORMER_ARITHMETIC_PACKAGE.json';m=json.loads(manifest.read_text());vendor=Path(m['vendor']);sources={}
 for name,rec in m['files'].items():
  f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256'];sources[str(f)]=rec['sha256']
 binding=GraphCall('ops-transformer')
 def run(x, weight):
         packed = x.contiguous().reshape(1, -1, 1, x.shape[-1])
         gamma = weight.contiguous()
         encoder = torch.empty((1, 0, 1, x.shape[-1]), device=x.device, dtype=x.dtype)
         zero = torch.zeros_like(gamma)
         one = torch.ones_like(gamma)
         rope_zero = torch.zeros((0, x.shape[-1]), device=x.device, dtype=x.dtype)
         outs = [torch.empty((1, 1, packed.shape[1], x.shape[-1]),
                             device=x.device, dtype=x.dtype) for _ in range(3)]
         binding.call("NormRopeConcat",
                      [packed, packed, packed, encoder, encoder, encoder, gamma, zero, gamma,
                       zero, one, zero, one, zero, rope_zero, rope_zero],
                      outs,
                      {"norm_type": 4, "norm_added_type": 0, "rope_type": 0,
                       "concat_order": 0, "eps": float(case.params["eps"]), "is_training": False})
         return outs[0].reshape(x.shape)
 for f in [Path(__file__),Path(__file__).with_name('ops_transformer_rms_api_graph_api.py'),q,manifest]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return Candidate(Implementation(run),'ops-transformer::original NormRopeConcat full3output RMS mode4','Originalnative runASTexact, full3outputs andallzero/one/encoder/ropeparameters timed; unchangedoriginal _adapt max_hidden1024.',sources,cleanup=binding.cleanup)
def candidate_factory(case,library):
 def build():
  from benchmarks.ascend_rms_cases import _adapt
  assert library=='ops-transformer' and case._call.signature.name=='RMSNormFwdOp'
  return _adapt(case,original_native,max_hidden=1024)
 return build
