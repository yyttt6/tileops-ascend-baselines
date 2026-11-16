from pathlib import Path
import json,hashlib
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-attention-v1')
def _gemm(*args,**kwargs):raise RuntimeError('Amax scope never invokes non-ReLU GEMM')
def _candidate(binding,run,op,directory,adaptation):
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 manifest=N/'TRANSFORMER_ATTENTION_PACKAGE.json';m=json.loads(manifest.read_text());sources={}
 for name,rec in m['files'].items():
  p=Path(m['vendor'])/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'];sources[str(p)]=rec['sha256']
 T=Path(__file__).parent/'workspace/TileOPs'
 for p in [Path(__file__),Path(__file__).with_name('ops_transformer_relu_api_graph_api.py'),manifest,T/'benchmarks/ascend_all_pairs/ops_transformer/extrema.py',T/'benchmarks/ascend_transformer_graph.py']:
  sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return Candidate(Implementation(run),'ops-transformer::original '+op,adaptation,sources,cleanup=binding.cleanup)
def candidate_factory(case,library):
 def build():
  assert library=='ops-transformer' and case._call.signature.name=='ReluFwdOp'
  assert json.loads(Path(__file__).with_name('TRANSFORMER_RELU_API_FORMAL_PREREQUISITES.json').read_text())['original950_and_both_original_guarded_hostbuilds_verified']
  return pointwise_extrema(case)
 return build


def pointwise_extrema(case):
    import torch
    from dataclasses import replace
    from ops_transformer_relu_api_graph_api import GraphCall
    from benchmarks.ascend_transformer_graph import _attention
    from benchmarks.ascend_baselines import UnsupportedCase, WORKSPACE
    from ops_transformer_relu_api_support import _candidate, _gemm
    from ops_transformer_relu_api_support import source_hashes
    dtype = case.inputs[0].dtype
    if dtype not in (torch.float16, torch.bfloat16) or any((torch.is_tensor(t) and t.dtype != dtype for t in case.inputs)):
        raise UnsupportedCase('Original attention extrema require matching FP16/BF16 inputs; no lossy narrowing.')
    op, p = (case._call.signature.name, case.params)
    for key in ('min', 'max', 'min_val', 'max_val'):
        bound = p.get(key)
        if isinstance(bound, (int, float)) and float(torch.tensor(bound, dtype=dtype).item()) != float(bound):
            raise UnsupportedCase('Scalar bound must be exactly representable in the original native input dtype.')
    binding = GraphCall('ops-transformer')

    def negate(x):
        weight = torch.full((1, 1), -1.0, dtype=x.dtype, device=x.device)
        out = _gemm(binding, x.reshape(-1, 1), weight, x.dtype).reshape(x.shape)
        binding.cleanup()
        return out

    def pair(a, b, minimum=False):
        a, b = torch.broadcast_tensors(a, b)
        packed = torch.stack((a, b), -1).reshape(-1, 2)
        chunks = []
        for start in range(0, packed.shape[0], 32768):
            rows = packed[start:start + 32768]
            keys = torch.nn.functional.pad(rows[:, :, None, None], (0, 127)).contiguous()
            query = torch.zeros((len(rows), 1, 1, 128), dtype=dtype, device=a.device)
            query[..., 0] = -1 if minimum else 1
            _, maximum = _attention(binding, query, keys, keys, 1.0)
            chunks.append(maximum[:, 0, 0, 0].to(dtype))
            binding.cleanup()
        result = torch.cat(chunks).reshape(a.shape)
        return negate(result) if minimum else result

    def run(x, low=None, high=None):
        if op in ('MaximumFwdOp', 'MinimumFwdOp'):
            return pair(x, low, op == 'MinimumFwdOp')
        if op == 'AbsFwdOp':
            return pair(x, negate(x))
        if op == 'ReluFwdOp':
            return pair(x, torch.zeros((), dtype=dtype, device=x.device))
        if op in ('ClampScalarFwdOp', 'HardtanhFwdOp'):
            lo = p.get('min_val') if op == 'HardtanhFwdOp' else p.get('min')
            hi = p.get('max_val') if op == 'HardtanhFwdOp' else p.get('max')
            low = None if lo is None else torch.full((), lo, dtype=dtype, device=x.device)
            high = None if hi is None else torch.full((), hi, dtype=dtype, device=x.device)
        result = pair(x, low) if low is not None else x
        return pair(result, high, True) if high is not None else result
    uses_negation = op in ('AbsFwdOp', 'MinimumFwdOp', 'HardtanhFwdOp') or (op == 'ClampScalarFwdOp' and p.get('max') is not None) or (op == 'ClampTensorFwdOp' and len(case.inputs) > 2 and (case.inputs[2] is not None))
    result = _candidate(binding, run, 'FlashAttentionScore' + ('+GroupedMatmulAdd' if uses_negation else ''), 'attention/flash_attention_score', 'Broadcast and pack two scalar keys; original attention raw maximum implements extrema. Signed basis queries and original constant -1 GEMM implement minima/absolute value. ReLU and clamps add exact constant/tensor bounds. All packing, native calls and bounded-lifetime synchronization are timed; no external floating comparisons, reductions or activation arithmetic.')
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__), WORKSPACE / 'ascend_baselines/ops-transformer/gmm/grouped_matmul_add'))

def source_hashes(*paths):
 result={}
 for value in paths:
  p=Path(value)
  if not p.exists() and 'ascend_baselines/ops-transformer/' in str(p):
   p=N/'source'/str(p).split('ascend_baselines/ops-transformer/',1)[1]
  assert p.exists(),str(p)
  files=[p] if p.is_file() else [f for f in p.rglob('*') if f.is_file() and f.suffix in ('.cpp','.h','.py','.json','.so','.o')]
  for f in files:result[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return result
