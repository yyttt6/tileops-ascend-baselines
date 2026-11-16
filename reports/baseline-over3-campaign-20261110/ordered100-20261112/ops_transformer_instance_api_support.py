from pathlib import Path
import json,hashlib
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-arithmetic-v1')
def _gemm(*args,**kwargs):raise RuntimeError('Amax scope never invokes minimum GEMM')
def _candidate(binding,run,op,directory,adaptation):
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 manifest=N/'TRANSFORMER_ARITHMETIC_PACKAGE.json';m=json.loads(manifest.read_text());sources={}
 for name,rec in m['files'].items():
  p=Path(m['vendor'])/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'];sources[str(p)]=rec['sha256']
 T=Path(__file__).parent/'workspace/TileOPs'
 for p in [Path(__file__),Path(__file__).with_name('ops_transformer_instance_api_graph_api.py'),manifest,T/'benchmarks/ascend_all_pairs/ops_transformer/adapters.py',T/'benchmarks/ascend_transformer_graph.py']:
  sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return Candidate(Implementation(run),'ops-transformer::original '+op,adaptation,sources,cleanup=binding.cleanup)
def candidate_factory(case,library):
 def build():
  assert library=='ops-transformer' and case._call.signature.name=='InstanceNormFwdOp'
  assert json.loads(Path(__file__).with_name('TRANSFORMER_INSTANCE_API_FORMAL_PREREQUISITES.json').read_text())['original950_and_both_original_guarded_hostbuilds_verified']
  return channel_norm(case)
 return build



def channel_norm(case):
    import math, torch
    from benchmarks.ascend_baselines import UnsupportedCase
    from ops_transformer_instance_api_graph_api import GraphCall
    p = case.params
    instance = case._call.signature.name == 'InstanceNormFwdOp'
    x = case.inputs[0]
    groups = x.shape[1] if instance else p['num_groups']
    spatial = math.prod(x.shape[2:])
    width = x.shape[1] // groups * spatial
    if width > 1024 or width % 2:
        raise UnsupportedCase('NormRopeConcat requires complete normalized width even and <=1024; splitting the reduction changes the contract.')
    if instance and p['use_input_stats'] and (case.inputs[1] is not None):
        raise UnsupportedCase('Original NormRopeConcat does not expose the complete running-mean/running-variance output contract.')
    if instance and (not p['use_input_stats']):
        raise UnsupportedCase('Original NormRopeConcat computes input statistics and cannot select supplied running mean/variance.')
    binding = GraphCall('ops-transformer')

    def run(x, *rest):
        weight, bias = rest[-2:]
        values = []
        channels = x.shape[1] // groups
        compute_dtype = torch.float32 if weight is not None or bias is not None else x.dtype
        for g in range(groups):
            inp = x[:, g * channels:(g + 1) * channels].to(compute_dtype).contiguous().reshape(1, x.shape[0], 1, width)
            gamma = torch.ones(width, dtype=compute_dtype, device=x.device) if weight is None else weight[g * channels:(g + 1) * channels, None].to(compute_dtype).expand(-1, spatial).contiguous().flatten()
            beta = torch.zeros_like(gamma) if bias is None else bias[g * channels:(g + 1) * channels, None].to(compute_dtype).expand(-1, spatial).contiguous().flatten()
            encoder = torch.empty((1, 0, 1, width), dtype=compute_dtype, device=x.device)
            rope = torch.empty((0, width), dtype=compute_dtype, device=x.device)
            one, zero = (torch.ones_like(gamma), torch.zeros_like(gamma))
            out = [torch.empty((1, 1, x.shape[0], width), dtype=compute_dtype, device=x.device) for _ in range(3)]
            binding.call('NormRopeConcat', [inp, inp, inp, encoder, encoder, encoder, gamma, beta, gamma, beta, one, zero, one, zero, rope, rope], out, {'norm_type': 2, 'norm_added_type': 0, 'rope_type': 0, 'concat_order': 0, 'eps': float(p['eps']), 'is_training': False})
            values.append(out[0].reshape(x.shape[0], channels, *x.shape[2:]))
        return torch.cat(values, 1).to(x.dtype)
    return _candidate(binding, run, 'NormRopeConcat', 'posembedding/norm_rope_concat', 'Group/instance normalization by complete per-group layer normalization; repeat channel affine parameters over spatial coordinates, original native mean/variance and epsilon, concatenate groups. Affine mode selects the original FP32 template by lossless input/parameter widening with a final output cast, all inside run.')

def source_hashes(*paths):
 return {str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
