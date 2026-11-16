from pathlib import Path
import json,hashlib
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-arithmetic-v1')
def _candidate(binding,run,op,directory,adaptation):
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 manifest=N/'TRANSFORMER_ARITHMETIC_PACKAGE.json';m=json.loads(manifest.read_text());sources={}
 for name,rec in m['files'].items():
  p=Path(m['vendor'])/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'];sources[str(p)]=rec['sha256']
 for p in [Path(__file__),Path(__file__).with_name('ops_transformer_coupling_api_graph_api.py'),Path(__file__).with_name('TRANSFORMER_COUPLING_API_FORMAL_PREREQUISITES.json'),Path(__file__).parent/'workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/adapters.py',manifest]:sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return Candidate(Implementation(run),'ops-transformer::original '+op,adaptation,sources,cleanup=binding.cleanup)
def candidate_factory(case,library):
 def build():
  assert library=='ops-transformer' and case._call.signature.name=='SSDChunkCouplingFwdOp'
  assert json.loads(Path(__file__).with_name('TRANSFORMER_COUPLING_API_FORMAL_PREREQUISITES.json').read_text())['original950_and_both_original_guarded_hostbuilds_verified']
  return coupling(case)
 return build

def _gemm(binding, a, b, out_dtype):
    import torch
    if not a.shape[0] or not a.shape[1]:
        return torch.zeros((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    out = torch.zeros((1, a.shape[0], b.shape[1]), dtype=torch.float32, device=a.device)
    group = torch.full((1,), a.shape[1], dtype=torch.int64, device=a.device)
    binding.call('GroupedMatmulAdd', [a.T.contiguous(), b.contiguous(), group, out], [out], {'transpose_x': True, 'transpose_weight': False, 'group_type': 2, 'group_list_type': 0})
    return out[0].to(out_dtype)
def coupling(case):
    import torch
    from ops_transformer_coupling_api_graph_api import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16):
        raise UnsupportedCase('Native coupling GEMM requires FP16/BF16; no FP32 narrowing.')
    binding = GraphCall('ops-transformer')
    chunk = case.params['chunk_len']

    def run(c, b):
        batch, s, groups, d = c.shape
        chunks = s // chunk
        cp = c.reshape(batch, chunks, chunk, groups, d).transpose(2, 3).contiguous().reshape(-1, chunk, d)
        bp = b.reshape(batch, chunks, chunk, groups, d).transpose(2, 3).contiguous().reshape(-1, chunk, d)
        outputs = []
        for lhs, rhs in zip(cp, bp, strict=True):
            dense = _gemm(binding, lhs, rhs.T, c.dtype)
            out = torch.zeros_like(dense)
            for row in range(chunk):
                out[row, :row + 1].copy_(dense[row, :row + 1])
            outputs.append(out)
        return torch.stack(outputs).reshape(batch, chunks, groups, chunk, chunk)
    return _candidate(binding, run, 'GroupedMatmulAdd', 'gmm/grouped_matmul_add', 'Chunk/group layout packing, original matrix product, causal row-prefix output cropping into a zero-padded output. Full multiplication and every copy are timed.')
