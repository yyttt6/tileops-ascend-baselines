"""Frozen LogSoftmax through complete original FP32 softmax and logarithm APIs."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))

def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='LogSoftmaxFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_all_pairs.tla import adapters,native
  from benchmarks.ascend_baselines import Candidate
  import tla_cache_artifact_identity as identity
  identity.install();native.prepare_deployment();x,=case.inputs
  assert x.dtype in (torch.float16,torch.bfloat16,torch.float32)
  shape=tuple(case.params['output'][0]);assert shape==tuple(x.shape)
  # The original online-softmax kernel already computes FP32; select its full
  # FP32 output API for the intermediate, avoiding a half-probability roundtrip.
  softparams={**case.params,'output':(shape,'float32')}
  soft=adapters.reduction('SoftmaxFwdOp',SimpleNamespace(params=softparams,inputs=(x,)))
  logparams={'input':(shape,'float32'),'output':case.params['output']}
  log=adapters.elementwise('LogFwdOp',SimpleNamespace(params=logparams,inputs=(x,)))
  children=(soft,log)
  assert all(c.implementation.args is None and c.implementation.reset is None for c in children)
  def run(value):
   probabilities=soft.implementation.run(value)
   return log.implementation.run(probabilities)
  def cleanup():
   for c in children:
    if c.cleanup is not None:c.cleanup()
  hashes={}
  for c in children:hashes.update(c.source_sha256)
  for f in (Path(identity.__file__),Path(__file__)):
   hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA original complete FP32 online_softmax + original complete ln','Frozen-case log(softmax(x)): original full axis permutation, contiguous FP32 widening, row padding, full online-softmax normalization/output restoration and FP32 intermediate; original complete logarithm pack/pad/native/crop and requested output cast. No kernel/block/passflags/checker changes. Full original checker must validate probability underflow/accuracy; not generic extreme-domain LogSoftmax acceptance.',hashes,cleanup=cleanup)
 return build
