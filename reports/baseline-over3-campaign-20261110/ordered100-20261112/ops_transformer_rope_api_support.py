from pathlib import Path
import json,hashlib
from ops_transformer_rope_api_graph_api import GraphCall
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-rope-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_rope_cases import _layout,_tables
  assert library=='ops-transformer' and case._call.signature.name=='RoPEFwdOp'
  batch,seq,heads,dim,neox=_layout(case)
  q=Path(__file__).with_name('TRANSFORMER_ROPE_API_FORMAL_PREREQUISITES.json');assert json.loads(q.read_text())['original950_and_both_original_guarded_hostbuilds_verified']
  manifest=N/'TRANSFORMER_ROPE_PACKAGE.json';m=json.loads(manifest.read_text());vendor=Path(m['vendor']);sources={}
  for name,rec in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256'];sources[str(f)]=rec['sha256']
  binding=GraphCall('ops-transformer')
  def run(x):
          tables = _tables(torch, x, case.params.get('base', 10000.))
          tables = [(torch.cat((t, t), dim=-1) if neox else t.repeat_interleave(2, dim=-1))
                    .float().reshape(1, seq, 1, dim) for t in tables]
          inp = x.float().contiguous().reshape(batch, seq, heads, dim)
          out = torch.empty_like(inp)
          binding.call('RotaryPositionEmbedding', [inp, *tables, None], [out], {'mode': 0 if neox else 1})
          return out.reshape_as(x).to(x.dtype)
  for f in [Path(__file__),Path(__file__).with_name('ops_transformer_rope_api_graph_api.py'),q,manifest]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'ops-transformer::original RotaryPositionEmbedding BSND','OriginalrunASTexact,_layout/_tables unchanged; original roundedtables/F32 widening/nativepairing/outputdtype/layout alltimed, no newmath orkernel.',sources,cleanup=binding.cleanup)
 return build
