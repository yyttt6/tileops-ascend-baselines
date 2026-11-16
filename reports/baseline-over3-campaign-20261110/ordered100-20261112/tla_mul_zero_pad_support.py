"""Scoped Mul wrapper: skip only existing zero-length pad; native kernel unchanged."""
from pathlib import Path
import sys,hashlib
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.ascend_all_pairs.tla import adapters
  from benchmarks.ascend_all_pairs.registry import candidate_factory as original
  import tla_zero_pad_candidate as shared
  import tla_cache_artifact_identity as identity
  assert library=='tilelang-ascend' and case._call.signature.name=='MulFwdOp' and case.id in ('hidden-state-prefill-float32','llama-7b-ffn-float32')
  assert len(case.inputs)==2 and all(x.dtype==torch.float32 and tuple(x.shape)==tuple(case.params['output'][0]) and x.numel()%4096==0 for x in case.inputs)
  identity.install();adapters._flat_pack=shared.clone_flat_pack(adapters);candidate=original(case,library)()
  for f in [Path(__file__),Path(shared.__file__),Path(identity.__file__)]:candidate.source_sha256[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return candidate
 return build
