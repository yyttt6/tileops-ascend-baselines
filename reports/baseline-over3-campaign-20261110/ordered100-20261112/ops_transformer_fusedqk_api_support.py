from pathlib import Path
import json,hashlib
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-arithmetic-v1')
def _gemm(binding, a, b, out_dtype):
    import torch
    if not a.shape[0] or not a.shape[1]:
        return torch.zeros((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    out = torch.zeros((1, a.shape[0], b.shape[1]), dtype=torch.float32, device=a.device)
    group = torch.full((1,), a.shape[1], dtype=torch.int64, device=a.device)
    binding.call('GroupedMatmulAdd', [a.T.contiguous(), b.contiguous(), group, out], [out], {'transpose_x': True, 'transpose_weight': False, 'group_type': 2, 'group_list_type': 0})
    return out[0].to(out_dtype)
def _candidate(binding,run,op,directory,adaptation):
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 G=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-rope-v1');gm=G/'TRANSFORMER_ROPE_PACKAGE.json';gd=json.loads(gm.read_text())
 for name,rec in gd['files'].items():
  fp=Path(gd['vendor'])/name;assert hashlib.sha256(fp.read_bytes()).hexdigest()==rec['sha256']
 manifest=N/'TRANSFORMER_ARITHMETIC_PACKAGE.json';m=json.loads(manifest.read_text());sources={}
 for name,rec in m['files'].items():
  p=Path(m['vendor'])/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'];sources[str(p)]=rec['sha256']
 T=Path(__file__).parent/'workspace/TileOPs'
 for p in [Path(__file__),Path(__file__).with_name('ops_transformer_fusedqk_api_graph_api.py'),manifest,T/'benchmarks/ascend_all_pairs/ops_transformer/adapters.py',T/'benchmarks/ascend_transformer_graph.py']:
  sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 sources[str(gm)]=hashlib.sha256(gm.read_bytes()).hexdigest()
 for name,rec in gd['files'].items():sources[str(Path(gd['vendor'])/name)]=rec['sha256']
 return Candidate(Implementation(run),'ops-transformer::original '+op,adaptation,sources,cleanup=binding.cleanup)
def candidate_factory(case,library):
 def build():
  assert library=='ops-transformer' and case._call.signature.name=='FusedQKNormRoPEFwdOp'
  assert json.loads(Path(__file__).with_name('TRANSFORMER_FUSEDQK_API_FORMAL_PREREQUISITES.json').read_text())['original950_and_both_original_guarded_hostbuilds_verified']
  return fused_qk_rope(case)
 return build



def fused_qk_rope(case):
    import torch
    from ops_transformer_fusedqk_api_graph_api import GraphCall
    from benchmarks.ascend_baselines import source_hashes, WORKSPACE
    from ops_transformer_fusedqk_api_support import source_hashes
    from dataclasses import replace
    binding = GraphCall('ops-transformer')
    p = case.params

    def run(qkv, qweight, kweight, tables, positions):
        width = qweight.numel()
        h = p['num_heads']
        g = p['num_kv_heads']
        result = qkv.clone().reshape(qkv.shape[0], h + 2 * g, width)
        cos, sin = tables[positions.long()].float().chunk(2, -1)
        rotary = cos.shape[-1] * 2
        coeff = [torch.cat((x, x), -1)[None, :, None, :].contiguous() for x in (cos, sin)]
        for start, count, weight in ((0, h, qweight), (h, g, kweight)):
            value = result[:, start:start + count].float().contiguous()[None]
            gamma = weight.float().contiguous()
            zero = torch.zeros_like(gamma)
            one = torch.ones_like(gamma)
            encoder = torch.empty((1, 0, count, width), dtype=torch.float32, device=qkv.device)
            rope_empty = torch.empty((0, width), dtype=torch.float32, device=qkv.device)
            outputs = [torch.empty((1, count, qkv.shape[0], width), dtype=torch.float32, device=qkv.device) for _ in range(3)]
            binding.call('NormRopeConcat', [value, value, value, encoder, encoder, encoder, gamma, zero, gamma, zero, one, zero, one, zero, rope_empty, rope_empty], outputs, {'norm_type': 4, 'norm_added_type': 0, 'rope_type': 0, 'concat_order': 0, 'eps': float(p['eps']), 'is_training': False})
            normalized = outputs[0].transpose(1, 2).contiguous()
            inp = normalized[..., :rotary].contiguous()
            inter = p['rope_layout'] == 'interleaved'
            if inter:
                inp = inp.reshape(*inp.shape[:-1], rotary // 2, 2).transpose(-1, -2).contiguous().reshape(inp.shape)
            rotated = torch.empty_like(inp)
            binding.call('RotaryPositionEmbedding', [inp, *coeff, None], [rotated], {'mode': 0})
            if inter:
                rotated = rotated.reshape(*rotated.shape[:-1], 2, rotary // 2).transpose(-1, -2).contiguous().reshape(inp.shape)
            normalized[..., :rotary].copy_(rotated)
            result[:, start:start + count].copy_(normalized[0].to(qkv.dtype))
        return result.reshape_as(qkv)
    result = _candidate(binding, run, 'NormRopeConcat+RotaryPositionEmbedding', 'posembedding/norm_rope_concat', 'Losslessly widen Q/K and weights to FP32, original affine RMS then original native RoPE with supplied tables, preserving FP32 intermediate contract; partial rotary crop and exact unchanged V columns. All transformations run per invocation.')
    return replace(result, source_sha256=result.source_sha256 | source_hashes(WORKSPACE / 'ascend_baselines/ops-transformer/posembedding/rotary_position_embedding'))

def source_hashes(*paths):
 result={}
 for value in paths:
  p=Path(value)
  if not p.exists() and 'ascend_baselines/ops-transformer/' in str(p):p=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-rope-v1')/'source'/str(p).split('ascend_baselines/ops-transformer/',1)[1]
  assert p.exists(),str(p)
  for f in ([p] if p.is_file() else [q for q in p.rglob('*') if q.is_file() and q.suffix in ('.h','.cpp','.py','.json','.so','.o')]):result[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return result
