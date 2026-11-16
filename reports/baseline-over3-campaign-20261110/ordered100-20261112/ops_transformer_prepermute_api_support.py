from pathlib import Path
import json,hashlib
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-routing-v1')
def _gemm(*args,**kwargs):raise RuntimeError('Amax scope never invokes minimum GEMM')
def _candidate(binding,run,op,directory,adaptation):
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 manifest=N/'TRANSFORMER_ROUTING_PACKAGE.json';m=json.loads(manifest.read_text());sources={}
 for name,rec in m['files'].items():
  p=Path(m['vendor'])/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'];sources[str(p)]=rec['sha256']
 T=Path(__file__).parent/'workspace/TileOPs'
 for p in [Path(__file__),Path(__file__).with_name('ops_transformer_prepermute_api_graph_api.py'),manifest,T/'benchmarks/ascend_transformer_graph.py',T/'benchmarks/ascend_transformer_graph.py']:
  sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return Candidate(Implementation(run),'ops-transformer::original '+op,adaptation,sources,cleanup=binding.cleanup)
def candidate_factory(case,library):
 def build():
  assert library=='ops-transformer' and case._call.signature.name=='MoEPrePermuteFwdOp'
  assert json.loads(Path(__file__).with_name('TRANSFORMER_PREPERMUTE_API_FORMAL_PREREQUISITES.json').read_text())['original950_and_both_original_guarded_hostbuilds_verified']
  return transformer_graph_prepermute(case)
 return build



from ops_transformer_prepermute_api_graph_api import GraphCall
from benchmarks.ascend_baselines import UnsupportedCase
def transformer_graph_prepermute(case):
    import torch
    if case.params['layout'].get('contiguous') != {'packing': 'tight', 'metadata_kind': 'physical_psum', 'alignment': 1}:
        raise UnsupportedCase('This pilot uses complete tight expert-major routing.')
    binding = GraphCall('ops-transformer')
    experts = case.params['num_local_experts']

    def run(x, ids):
        routes = ids.numel()
        out = torch.empty((routes, x.shape[1]), device=x.device, dtype=x.dtype)
        inverse = torch.empty(routes, device=x.device, dtype=torch.int32)
        prefix = torch.empty(experts, device=x.device, dtype=torch.int32)
        binding.call('MoeInitRoutingV2', [x.contiguous(), ids.contiguous()], [out, inverse, prefix], {'active_num': routes, 'expert_capacity': 0, 'expert_num': experts, 'drop_pad_mode': 0, 'expert_tokens_count_or_cumsum_flag': 1, 'expert_tokens_before_capacity_flag': False})
        return (out, prefix, inverse)
    return _candidate(binding, run, 'MoeInitRoutingV2', 'moe/moe_init_routing_v2', 'Native dropless routing, prefix counts and original inverse indices; all outputs checked.')
