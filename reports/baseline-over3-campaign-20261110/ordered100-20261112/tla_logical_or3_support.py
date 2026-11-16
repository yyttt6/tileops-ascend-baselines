"""Finite floating logical route: two full original Ne APIs and one full original bitwise API."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='LogicalOrFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  import tla_unpack_compat_v3 as unpack
  identity.install();native.prepare_deployment();a,b=case.inputs
  assert a.dtype==b.dtype and a.dtype in (torch.float16,torch.bfloat16,torch.float32)
  shape=tuple(case.params['output'][0]);assert tuple(torch.broadcast_shapes(a.shape,b.shape))==shape
  dt=str(a.dtype).split('.')[-1]
  def descriptor(dtype,xshape,yshape):
   return SimpleNamespace(params={'input':(tuple(xshape),dtype),'other':(tuple(yshape),dtype),'output':(shape,'bool')},inputs=(a,b))
  compare=unpack.clone(adapters)
  left=compare('NeFwdOp',descriptor(dt,a.shape,a.shape));right=compare('NeFwdOp',descriptor(dt,b.shape,b.shape));combine=adapters.elementwise('BitwiseOrFwdOp',descriptor('bool',shape,shape));children=(left,right,combine)
  assert all(c.implementation.args is None and c.implementation.reset is None for c in children)
  def run(x,y):
   # Every zero, predicate, broadcast/pack/cast/unpack and bitwise API stays in this call.
   tx=left.implementation.run(x,torch.zeros_like(x))
   ty=right.implementation.run(y,torch.zeros_like(y))
   return combine.implementation.run(tx,ty)
  def cleanup():
   for c in children:
    if c.cleanup is not None:c.cleanup()
  hashes={}
  for c in children:hashes.update(c.source_sha256)
  for module in [identity,unpack]:
   f=Path(module.__file__);hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  hashes[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA original Ne + original Ne + original bitwise_or (three complete APIs)','Two per-call full zero tensors; original Ne predicates including full original FP32 widening/pack/byte-unpack; exact BOOL-to-INT16 original bitwise API and original cast/crop. No torch predicate, kernel, block, pass_config or precision edits. Selected API composition, not old unsupported provider timing.',hashes,cleanup=cleanup)
 return build
