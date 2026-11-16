"""Frozen Mean via complete original full-axis Sum and scalar-multiply APIs."""
from pathlib import Path
from types import SimpleNamespace
from math import prod
import hashlib,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))

def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='MeanFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  import tla_hardsigmoid_support as original_scalar
  identity.install();native.prepare_deployment();x,=case.inputs
  assert x.dtype in (torch.float16,torch.bfloat16,torch.float32)
  dims=case.params.get('dim');dims=tuple(range(x.ndim)) if dims is None else (dims,) if isinstance(dims,int) else tuple(dims)
  width=prod(x.shape[d%x.ndim] for d in dims)
  assert 0<width<=32768,'Original whole-row width bound retained'
  shape=tuple(case.params['output'][0]);outdtype=getattr(torch,case.params['output'][1])
  sump={**case.params,'x':(tuple(x.shape),'float32'),'output':(shape,'float32'),'dtype':'float32'}
  summation=adapters.reduction('SumFwdOp',SimpleNamespace(params=sump,inputs=(x,)))
  scalarp={'input':(shape,'float32'),'output':(shape,'float32')}
  scalar=original_scalar.scalar_factory(adapters,'vec_muls',1./width)('NegFwdOp',SimpleNamespace(params=scalarp,inputs=(x,)))
  children=(summation,scalar)
  assert all(c.implementation.args is None and c.implementation.reset is None for c in children)
  def run(value):
   total=summation.implementation.run(value)
   return scalar.implementation.run(total).to(outdtype)
  def cleanup():
   for c in children:
    if c.cleanup is not None:c.cleanup()
  hashes={}
  for c in children:hashes.update(c.source_sha256)
  for f in (Path(identity.__file__),Path(original_scalar.__file__),Path(__file__)):
   hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA original complete full-axis sum + original complete vec_muls(1/width)','Frozen Mean: preserve every reduction axis and logical divisor, original whole-row width bound and row-only padding. Complete original FP32 Sum row packing/native calls/cat/output restoration, complete scalar API repacking/pad/native/crop, FP32 intermediate and requested output cast timed. No kernel/block/passflags/checker changes; complete original checker determines frozen-case numerical acceptance.',hashes,cleanup=cleanup)
 return build
