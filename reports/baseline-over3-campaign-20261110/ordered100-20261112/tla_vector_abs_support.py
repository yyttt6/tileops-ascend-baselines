"""Exact frozen FP32 L1/infinity via two complete original TLA APIs."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,sys,math
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='VectorNormFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  identity.install();native.prepare_deployment();x,=case.inputs;p=case.params;order=p['ord']
  assert x.dtype==torch.float32 and p['output'][1]=='float32' and p['dtype'] is None and order in (1,float('inf'))
  assert case.id==('l1-mlp-intermediate-float32' if order==1 else 'inf-mlp-intermediate-float32')
  shape=tuple(x.shape);absolute=adapters.elementwise('AbsFwdOp',SimpleNamespace(params={'input':(shape,'float32'),'output':(shape,'float32')},inputs=(x,)))
  op='SumFwdOp' if order==1 else 'AmaxFwdOp';reduce=adapters.reduction(op,SimpleNamespace(params={**p,'x':(shape,'float32'),'dtype':'float32'},inputs=(x,)))
  def run(value):return reduce.implementation.run(absolute.implementation.run(value))
  def cleanup():
   for c in (absolute,reduce):
    if c.cleanup:c.cleanup()
  hashes={**absolute.source_sha256,**reduce.source_sha256}
  for m in [identity]:hashes[str(Path(m.__file__))]=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
  hashes[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA complete original Abs + full-axis '+op,'ExactFP32 originalinputs/output. All full originalpacking/padding/stripcalls/cat/outputcrop counted; original widthbound/checker/precision unchanged. No newkernel/block/flags.',hashes,cleanup=cleanup)
 return build
