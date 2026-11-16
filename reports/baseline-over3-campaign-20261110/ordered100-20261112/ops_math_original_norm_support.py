"""Complete pinned original LayerNorm API; original affine/row/run AST retained."""
from pathlib import Path
import json,hashlib,os
from ops_math_original_norm_layout import norm_layout_run
P=Path(__file__).resolve().parent;N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  assert library=='ops-math';op=case._call.signature.name;assert op in {'LayerNormFwdOp','AdaLayerNormFwdOp','GroupNormFwdOp','InstanceNormFwdOp'}
  qf=N/'ORIGINAL_NORM_HOST_QUALIFICATION.json';q=json.loads(qf.read_text());assert q['full_API_embedded_kernel_source_build_guard_and_dual_archive_verified'] and q['execution_hostname']==os.uname().nodename
  for name,h in q['formal_source_sha256'].items():assert sha(P/name)==h
  bf=N/'BUILD_RECEIPT.json';binary=N/'libops_math_layernorm.so';src=N/'layer_norm.asc';assert sha(binary)==q['build_record']['binary_sha256'] and sha(src)==q['original_source_sha256']
  torch.ops.load_library(str(binary));native=torch.ops.allpairs_math_example.layer_norm;run=norm_layout_run(case,op,native)
  sources={str(f):sha(f) for f in [qf,bf,binary,src,P/'ops_math_original_norm_layout.py',P/'ops_math_original_norm_support.py',P/'workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/layernorm.py']}
  return Candidate(Implementation(run),'ops-math pinned complete native layer_norm plus original affine/row/run AST','Original source, full FP32 row conversion, packing, affine and final cast retained inside timed run',sources)
 return build
