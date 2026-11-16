from pathlib import Path
import json,hashlib
from ops_transformer_add_api_graph_api import GraphCall
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-arithmetic-v1')
def weighted_rows(binding, values, weights):
    """values[N,K,H], weights[N,K]; only address/layout construction here."""
    import torch
    n, k, h = values.shape
    packed = values.contiguous().reshape(n * k, h)
    ids = torch.arange(n * k, dtype=torch.int32, device=values.device).reshape(n, k).T.contiguous().flatten()
    out = torch.empty((n, h), dtype=values.dtype, device=values.device)
    binding.call('MoeFinalizeRoutingV2', [packed, ids, None, None, None, weights.contiguous(), None],
                 [out], {'drop_pad_mode': 0, 'k': k})
    return out
def candidate_factory(case,library):
 def build():
  import torch,math
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  op,p=case._call.signature.name,case.params
  assert library=='ops-transformer' and op=='AddFwdOp';assert case.inputs[0].dtype in (torch.float32,torch.bfloat16);assert p.get('alpha',1)==1
  q=Path(__file__).with_name('TRANSFORMER_ADD_API_FORMAL_PREREQUISITES.json');assert json.loads(q.read_text())['original950_and_both_original_guarded_hostbuilds_verified']
  manifest=N/'TRANSFORMER_ARITHMETIC_PACKAGE.json';m=json.loads(manifest.read_text());vendor=Path(m['vendor']);sources={}
  for name,rec in m['files'].items():
   f=vendor/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256'];sources[str(f)]=rec['sha256']
  binding=GraphCall('ops-transformer')
  def run(x, y=None):
          if op in ('SumFwdOp', 'MeanFwdOp'):
              dims = p['dim']
              dims = range(x.ndim) if dims is None else ((dims,) if isinstance(dims, int) else dims)
              dims = sorted(d % x.ndim for d in dims)
              kept = [i for i in range(x.ndim) if i not in dims]
              width = math.prod(x.shape[i] for i in dims)
              values = x.permute(kept + dims).contiguous().reshape(-1, width, 1)
              weights = torch.full(values.shape[:2], 1 / width if op == 'MeanFwdOp' else 1., dtype=x.dtype, device=x.device)
              out = weighted_rows(binding, values, weights)
              shape = [1 if i in dims else size for i, size in enumerate(x.shape)] if p['keepdim'] else [x.shape[i] for i in kept]
              dtype = getattr(torch, p['dtype']) if p.get('dtype') else x.dtype
              return out.to(dtype).reshape(shape)
          if y is None:
              values = x.contiguous().reshape(-1, 1, 1)
              weights = torch.full(values.shape[:2], -1., dtype=x.dtype, device=x.device)
              return weighted_rows(binding, values, weights).reshape_as(x)
          a, b = torch.broadcast_tensors(x, y)
          if op == 'AddFwdOp':
              # The original x1 path copies the accumulator instead of adding
              # a leading +0. This preserves -0 + -0 for the unchanged edge oracle.
              lhs, rhs = a.float().contiguous().reshape(-1, 1), b.float().contiguous().reshape(-1, 1)
              ids = torch.arange(len(lhs), dtype=torch.int32, device=x.device)
              out = torch.empty_like(lhs)
              binding.call('MoeFinalizeRoutingV2', [rhs, ids, lhs, None, None, None, None],
                           [out], {'drop_pad_mode': 0, 'k': 1})
              return out.reshape(a.shape).to(x.dtype)
          if op == 'MulFwdOp':
              values = a.contiguous().reshape(-1, 1, 1)
              weights = b.contiguous().reshape(-1, 1)
          else:
              values = torch.stack((a, b), -1).float().reshape(-1, 2, 1)
              w = p.get('weight')
              coefficients = (1 - w, w) if op == 'LerpScalarFwdOp' else ((1., 1.) if op == 'AddFwdOp' else (1., -1.))
              weights = torch.tensor(coefficients, dtype=torch.float32, device=x.device).expand(values.shape[0], -1).contiguous()
          return weighted_rows(binding, values, weights).reshape(a.shape).to(x.dtype)
  for f in [Path(__file__),Path(__file__).with_name('ops_transformer_add_api_graph_api.py'),q,manifest]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'ops-transformer::original MoeFinalizeRoutingV2 accumulatorcopy+expert Add','Exact original routing_arithmetic.arithmetic runAST; originalF32 widening forBF16,completeAPI+casts/broadcast/metadata, no newmath/kernel.',sources,cleanup=binding.cleanup)
 return build
