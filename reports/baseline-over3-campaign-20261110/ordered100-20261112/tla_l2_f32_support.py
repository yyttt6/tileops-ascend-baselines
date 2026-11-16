"""Exact FP32 L2 case through three complete original TLA APIs; no new kernel."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='VectorNormFwdOp' and case.id=='l2-mlp-intermediate-float32'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  identity.install();native.prepare_deployment();x,=case.inputs;p=case.params
  assert x.dtype==torch.float32 and p.get('ord') in (2,2.0) and p['output'][1]=='float32'
  shape=tuple(x.shape);out=tuple(p['output'][0]);param={'input':(shape,'float32'),'other':(shape,'float32'),'output':(shape,'float32')}
  mul=adapters.elementwise('MulFwdOp',SimpleNamespace(params=param,inputs=(x,x)))
  reduce=adapters.reduction('SumFwdOp',SimpleNamespace(params={**p,'x':(shape,'float32'),'dtype':'float32'},inputs=(x,)))
  sqrt=adapters.elementwise('SqrtFwdOp',SimpleNamespace(params={'input':(out,'float32'),'output':(out,'float32')},inputs=(x,)))
  def run(value):
   return sqrt.implementation.run(reduce.implementation.run(mul.implementation.run(value,value)))
  def cleanup():
   for c in (mul,reduce,sqrt):
    if c.cleanup:c.cleanup()
  hashes={**mul.source_sha256,**reduce.source_sha256,**sqrt.source_sha256}
  for m in [identity]:hashes[str(Path(m.__file__))]=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
  hashes[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA complete original Mul + whole-axis Sum + Sqrt','Only exact FP32 L2 frozen case. Full original flatten/pad/copy/native/outputcrop, complete reduction strip calls/cat and all FP32 intermediates timed. Original widthbound and checker unchanged; no manual precision promotion/newkernel/block/compilerflags.',hashes,cleanup=cleanup)
 return build
