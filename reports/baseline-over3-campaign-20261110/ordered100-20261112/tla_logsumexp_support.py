"""Finite frozen LogSumExp cases through three complete original native APIs."""
from pathlib import Path
from types import SimpleNamespace
from math import prod
import hashlib,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))

def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='LogSumExpFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  identity.install();native.prepare_deployment();x,=case.inputs
  assert x.dtype in (torch.float16,torch.bfloat16,torch.float32)
  shape=tuple(x.shape);outshape=tuple(case.params['output'][0]);dims=case.params['dim']
  dims=(dims,) if isinstance(dims,int) else tuple(dims)
  assert prod(shape[d%len(shape)] for d in dims)<=32768,'Original whole-row reduction bound retained'
  expdesc=SimpleNamespace(params={'input':(shape,'float32'),'output':(shape,'float32')},inputs=(x,))
  exp=adapters.elementwise('ExpFwdOp',expdesc)
  sump={**case.params,'x':(shape,'float32'),'output':(outshape,'float32'),'dtype':'float32'}
  summation=adapters.reduction('SumFwdOp',SimpleNamespace(params=sump,inputs=(x,)))
  logp={'input':(outshape,'float32'),'output':case.params['output']}
  log=adapters.elementwise('LogFwdOp',SimpleNamespace(params=logp,inputs=(x,)))
  children=(exp,summation,log)
  assert all(c.implementation.args is None and c.implementation.reset is None for c in children)
  def run(value):
   exponentials=exp.implementation.run(value)
   total=summation.implementation.run(exponentials)
   return log.implementation.run(total)
  def cleanup():
   for c in children:
    if c.cleanup is not None:c.cleanup()
  hashes={}
  for c in children:hashes.update(c.source_sha256)
  for f in (Path(identity.__file__),Path(__file__)):
   hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA original complete exp + original complete full-axis sum + original complete ln','Frozen-case log(sum(exp(x))): original complete FP32 exponential, all reduction axes and original whole-row width bound, full row packing/padding/native calls/cat/shape restoration, FP32 intermediate tensors and original output cast timed. No kernel/block/passflags/checker edits. Full original checker must validate overflow/accuracy; not generic stable or extreme-domain LogSumExp acceptance.',hashes,cleanup=cleanup)
 return build
