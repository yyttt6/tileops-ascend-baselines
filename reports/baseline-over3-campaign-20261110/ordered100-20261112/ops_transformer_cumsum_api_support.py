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
 for p in [Path(__file__),Path(__file__).with_name('ops_transformer_cumsum_api_graph_api.py'),manifest,T/'benchmarks/ascend_all_pairs/ops_transformer/prefix_sum.py',T/'benchmarks/ascend_transformer_graph.py']:
  sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return Candidate(Implementation(run),'ops-transformer::original '+op,adaptation,sources,cleanup=binding.cleanup)
def candidate_factory(case,library):
 def build():
  assert library=='ops-transformer' and case._call.signature.name=='CumsumFwdOp'
  assert json.loads(Path(__file__).with_name('TRANSFORMER_CUMSUM_API_FORMAL_PREREQUISITES.json').read_text())['original950_and_both_original_guarded_hostbuilds_verified']
  return prefix_sum(case)
 return build


def prefix_sum(case):
    import torch
    from dataclasses import replace
    from ops_transformer_cumsum_api_graph_api import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    from ops_transformer_cumsum_api_support import _candidate
    from ops_transformer_cumsum_api_support import source_hashes
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16):
        raise UnsupportedCase('Original GroupedMatmulAdd inputs are FP16/BF16; FP32 or integer narrowing is forbidden.')
    binding = GraphCall('ops-transformer')
    dim = case.params['dim']

    def run(x):
        packed = x.movedim(dim, -1).contiguous()
        width = packed.shape[-1]
        rows = packed.reshape(-1, width)
        carry = torch.zeros((len(rows), 1), dtype=torch.float32, device=x.device)
        parts = []
        for start in range(0, width, 128):
            block = rows[:, start:start + 128].contiguous()
            count = block.shape[1]
            triangular = torch.ones((count, count), dtype=x.dtype, device=x.device).triu()
            out = carry.expand(len(rows), count).clone()[None]
            group = torch.full((1,), count, dtype=torch.int64, device=x.device)
            binding.call('GroupedMatmulAdd', [block.T.contiguous(), triangular, group, out], [out], {'transpose_x': True, 'transpose_weight': False, 'group_type': 2, 'group_list_type': 0})
            carry = out[0, :, -1:].clone()
            parts.append(out[0].to(x.dtype))
            binding.cleanup()
        return torch.cat(parts, -1).reshape(packed.shape).movedim(-1, dim).contiguous()
    result = _candidate(binding, run, 'GroupedMatmulAdd', 'gmm/grouped_matmul_add', 'Move scan axis into native GEMM K; exact upper-triangular zero/one constants produce each inclusive prefix. Carry original FP32 final output as the next native accumulator before output casting. Complete axis packing, copies, native calls and synchronization are timed; no Torch cumulative arithmetic.')
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__)))

def source_hashes(*paths):
 return {str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
