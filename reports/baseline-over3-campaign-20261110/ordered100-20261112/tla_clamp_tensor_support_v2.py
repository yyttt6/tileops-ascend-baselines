"""Finite ClampTensor route through complete original maximum/minimum APIs."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))

def candidate_factory(case, library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='ClampTensorFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  identity.install();native.prepare_deployment()
  x,*bounds=case.inputs;lo=bounds[0] if bounds else None;hi=bounds[1] if len(bounds)>1 else None
  assert len(bounds)<=2 and (lo is not None or hi is not None)
  assert x.dtype in (torch.float16,torch.bfloat16,torch.float32)
  shape=tuple(case.params['output'][0]);dt=str(x.dtype).split('.')[-1]
  assert case.params['output'][1]==dt
  def child(op,bound):
   if bound is None:return None
   assert isinstance(bound,torch.Tensor) and bound.dtype==x.dtype
   assert tuple(torch.broadcast_shapes(x.shape,bound.shape))==shape
   descriptor=SimpleNamespace(params={'input':(tuple(x.shape),dt),'other':(tuple(bound.shape),dt),'output':(shape,dt)},inputs=(x,bound))
   return adapters.elementwise(op,descriptor)
  lower=child('MaximumFwdOp',lo);upper=child('MinimumFwdOp',hi)
  children=tuple(c for c in (lower,upper) if c is not None)
  assert all(c.implementation.args is None and c.implementation.reset is None for c in children)
  def run(value,minimum=None,maximum=None):
   assert (minimum is None)==(lower is None) and (maximum is None)==(upper is None)
   if lower is not None:value=lower.implementation.run(value,minimum)
   if upper is not None:value=upper.implementation.run(value,maximum)
   return value
  def cleanup():
   for c in children:
    if c.cleanup is not None:c.cleanup()
  hashes={}
  for c in children:hashes.update(c.source_sha256)
  for f in (Path(identity.__file__),Path(__file__)):
   hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA original maximum then original minimum (one or two complete APIs)','Original optional tensor bounds and dtype; all per-call broadcast/contiguous/FP32 widening/64x64 pad/native/crop/output casts retained, including intermediate materialization. Selected composition, not old unsupported provider timing.',hashes,cleanup=cleanup)
 return build
