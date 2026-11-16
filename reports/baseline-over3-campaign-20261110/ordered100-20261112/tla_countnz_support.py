"""Floating Any via full original Ne and full original CountNonzero APIs."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='CountNonzeroFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  import tla_unpack_compat_v3 as unpack
  identity.install();native.prepare_deployment();x,=case.inputs;shape=tuple(x.shape);dt=str(x.dtype).split('.')[-1];assert dt in ('float16','bfloat16','float32')
  compare=unpack.clone(adapters)('NeFwdOp',SimpleNamespace(params={'input':(shape,dt),'other':(shape,dt),'output':(shape,'bool')},inputs=(x,x)))
  reduce=adapters.reduction('SumFwdOp',SimpleNamespace(params={**case.params,'x':(shape,'float32'),'dtype':'float32'},inputs=(x,)))
  def run(value):
   mask=compare.implementation.run(value,torch.zeros_like(value))
   return reduce.implementation.run(mask)
  def cleanup():
   for c in (compare,reduce):
    if c.cleanup:c.cleanup()
  hashes={**compare.source_sha256,**reduce.source_sha256}
  for m in [identity,unpack]:hashes[str(Path(m.__file__))]=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
  hashes[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA original complete Ne + original complete full-axis Sum','Every per-call zero allocation, full Ne widening/pack/pad/native/crop/byteunpack, BOOL intermediate, exact FP32 zero/one widening and original full-axis rowpack/rowpad/stripcalls/cat/outputcast retained. No kernel/block/checker/precision edits; original widthbound retained.',hashes,cleanup=cleanup)
 return build
