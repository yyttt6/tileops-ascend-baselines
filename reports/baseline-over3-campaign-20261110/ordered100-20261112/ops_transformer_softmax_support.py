"""Frozen wide-axis original full-softmax-result API body, exact FP32 widening unchanged."""
from pathlib import Path
import json,hashlib
from ops_transformer_softmax_graph_api import GraphCall
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-selection-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  assert library=='ops-transformer' and case._call.signature.name=='SoftmaxFwdOp'
  x,p=case.inputs[0],case.params;dim=p['dim']%x.ndim;assert x.shape[dim]>1024
  dtype=getattr(torch,p['dtype']) if p.get('dtype') else x.dtype
  assert x.dtype in (torch.float16,torch.bfloat16,torch.float32) and dtype in (x.dtype,torch.float32)
  q=Path(__file__).with_name('TRANSFORMER_SOFTMAX_FORMAL_PREREQUISITES.json');assert json.loads(q.read_text())['complete3output_fixture_and_private_source_launch_verified_bothhosts']
  manifest=N/'TRANSFORMER_SELECTION_PACKAGE.json';m=json.loads(manifest.read_text());vendor=Path(m['vendor']);sources={}
  for name,rec in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256'];sources[str(f)]=rec['sha256']
  binding=GraphCall('ops-transformer',vendor_name='selectionpairs_transformer')
  def run(x):
        moved = x.movedim(dim, -1)
        width = moved.shape[-1]
        rows = moved.float().contiguous().reshape(-1, width)
        outputs = []
        for begin in range(0, len(rows), 128):
            inp = rows[begin:begin + 128]
            selected = torch.empty((len(inp), 1), dtype=torch.float32, device=x.device)
            ids = torch.empty_like(selected, dtype=torch.int32)
            full = torch.empty_like(inp)
            binding.call('MoeGatingTopKSoftmaxV2', [inp, None], [selected, ids, full],
                         {'k': 1, 'renorm': 0, 'output_softmax_result_flag': True})
            outputs.append(full)
            binding.cleanup()
        return torch.cat(outputs).reshape(moved.shape).movedim(-1, dim).to(dtype).contiguous()
  for f in [Path(__file__),Path(__file__).with_name('ops_transformer_softmax_graph_api.py'),q,manifest]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'ops-transformer::original MoeGatingTopKSoftmaxV2 full3output fullaxis API via source-owned graphJIT','Exact original wide-softmax timed body, includes all3output work, layouts/casts/descriptor/cleanup; no external softmax or axis splitting.',sources,cleanup=binding.cleanup)
 return build
