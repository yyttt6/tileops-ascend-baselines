"""Frozen channel-wise PReLU through three complete original native APIs."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))

def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='PreluFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  import tla_hardswish_support as original_fma
  identity.install();native.prepare_deployment();x,w=case.inputs
  assert x.dtype==w.dtype and x.dtype in (torch.float16,torch.bfloat16)
  assert w.ndim==1 and x.ndim>=2 and w.numel() in (1,x.shape[1])
  shape=tuple(case.params['output'][0]);assert shape==tuple(x.shape)
  weightshape=(1,w.numel())+(1,)*(x.ndim-2)
  params={'input':(shape,'float32'),'other':(shape,'float32'),'output':(shape,'float32')}
  desc=SimpleNamespace(params=params,inputs=(x,x))
  pos=adapters.elementwise('MaximumFwdOp',desc)
  neg=adapters.elementwise('MinimumFwdOp',desc)
  fmadesc=SimpleNamespace(params={**params,'other':(weightshape,'float32')},inputs=(x,w,x))
  fma=original_fma.fma_factory(adapters)('NegFwdOp',fmadesc)
  children=(pos,neg,fma)
  assert all(c.implementation.args is None and c.implementation.reset is None for c in children)
  def run(value,weight):
   zero=torch.zeros_like(value,dtype=torch.float32)
   positive=pos.implementation.run(value,zero)
   negative=neg.implementation.run(value,zero)
   weights=weight.reshape(weightshape)
   return fma.implementation.run(negative,weights,positive).to(value.dtype)
  def cleanup():
   for c in children:
    if c.cleanup is not None:c.cleanup()
  hashes={}
  for c in children:hashes.update(c.source_sha256)
  for f in (Path(identity.__file__),Path(original_fma.__file__),Path(__file__)):
   hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA original complete maximum + minimum + vec_mul_add_dst','Frozen F16/BF16 channel-wise PReLU: max(x,0)+weight*min(x,0), complete original FP32 max/min and ternary FMA. Per-call full zero tensor, channel weight view/full native broadcast materialization, all pack/pad/crop/intermediates and output cast timed. No predicate replacement, kernel/block/passflags/precision/checker edits; numerical acceptance requires original full checker.',hashes,cleanup=cleanup)
 return build
