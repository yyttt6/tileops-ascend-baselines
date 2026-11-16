from pathlib import Path
import json,hashlib
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-chunk-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.ascend_all_pairs.fla.cube_matmul import gemm
  from benchmarks.ascend_all_pairs.fla import PROVIDERS
  assert library=='flash-linear-attention-npu' and case._call.signature.name=='GemmFwdOp' and case.inputs[0].dtype==torch.bfloat16
  assert json.loads(Path(__file__).with_name('FLA_CUBE_API_FORMAL_PREREQUISITES.json').read_text())['original950_and_both_original_guarded_hostbuilds_verified']
  manifest=N/'PACKAGE_MANIFEST.json';m=json.loads(manifest.read_text());sources={}
  for name,h in m['artifact_sha256'].items():
   p=Path(m['package'])/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;sources[str(p)]=h
  candidate=PROVIDERS['GemmFwdOp'](case)
  assert candidate.api=='FLA npu_chunk_fwd_h(W=0,g=0, final FP32 state)',candidate.api
  assert candidate.implementation.run.__code__.co_filename.endswith('/fla/cube_matmul.py')
  sources.update(candidate.source_sha256)
  for p in [Path(__file__),manifest,Path(__file__).with_name('FLA_CUBE_API_FORMAL_PREREQUISITES.json')]:sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  candidate.source_sha256.update(sources)
  return candidate
 return build
