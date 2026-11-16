from pathlib import Path
import json,hashlib
from ops_transformer_softmax_narrow_graph_api import GraphCall
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-narrow-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  assert library=='ops-transformer' and case._call.signature.name=='SoftmaxFwdOp'
  p=case.params;x=case.inputs[0];dim=p['dim']%x.ndim;width=x.shape[dim];assert width<=1024
  dtype=getattr(torch,p['dtype']) if p.get('dtype') else x.dtype;assert dtype in (x.dtype,torch.float32)
  q=Path(__file__).with_name('TRANSFORMER_SOFTMAX_NARROW_FORMAL_PREREQUISITES.json');assert json.loads(q.read_text())['original3output_fixture_passed_H2']
  manifest=N/'TRANSFORMER_NARROW_PACKAGE.json';m=json.loads(manifest.read_text());vendor=Path(m['vendor']);sources={}
  for name,rec in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256'];sources[str(f)]=rec['sha256']
  binding=GraphCall('ops-transformer')
  def run(x):
          moved=x.movedim(dim,-1);packed=moved.to(dtype).contiguous().reshape(-1,width)
          y=torch.empty_like(packed);idx=torch.empty(packed.shape,dtype=torch.int32,device=x.device);rows=torch.empty_like(idx)
          binding.call('MoeGatingTopKSoftmax',[packed,None],[y,idx,rows],{'k':width})
          out=torch.empty_like(y).scatter_(1,idx.long(),y)
          return out.reshape(moved.shape).movedim(-1,dim).contiguous()
  for f in [Path(__file__),Path(__file__).with_name('ops_transformer_softmax_narrow_graph_api.py'),q,manifest]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'ops-transformer::original MoeGatingTopKSoftmax full3output k=width permutation','Original narrowSoftmax timedrun copiedexactAST; all3output API/cast/scatter/layout work preserved, no newmath/kernel.',sources,cleanup=binding.cleanup)
 return build
