"""Exact FP32 AdaLN/Zero through complete original TLA normalization/FMA/Mul APIs."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name in ('AdaLayerNormFwdOp','AdaLayerNormZeroFwdOp') and case.id=='dit-xl-2-float32'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  import tla_hardswish_support as fma_support
  import tla_direct_norm_selector as direct
  proof=P/'TLA_DIRECT_NORM_QUALIFIED.json';assert __import__('json').loads(proof.read_text())['direct_original_API_qualified'] is True
  identity.install();native.prepare_deployment();p=case.params;shape=tuple(p['x'][0]);zero=case._call.signature.name=='AdaLayerNormZeroFwdOp';assert len(case.inputs)==(4 if zero else 3) and all(v.dtype==torch.float32 and tuple(v.shape)==shape for v in case.inputs);assert p['output'][1]=='float32' and len(shape)==2
  x=case.inputs[0];norm=direct.select_direct(adapters)('LayerNormFwdOp',SimpleNamespace(params={'x':(shape,'float32'),'output':(shape,'float32'),'normalized_shape':(shape[-1],),'eps':p['eps']},inputs=(x,)))
  params={'input':(shape,'float32'),'other':(shape,'float32'),'output':(shape,'float32')}
  affine=fma_support.fma_factory(adapters)('NegFwdOp',SimpleNamespace(params=params,inputs=(x,x,x)))
  gate=adapters.elementwise('MulFwdOp',SimpleNamespace(params=params,inputs=(x,x))) if zero else None
  def run(value,scale,shift,*gating):
   normed=norm.implementation.run(value.float())
   out=affine.implementation.run(scale.float(),normed,shift.float())
   return (gate.implementation.run(out,gating[0].float()) if zero else out).to(value.dtype)
  def cleanup():
   for c in (norm,affine,gate):
    if c is not None and c.cleanup:c.cleanup()
  hashes={**norm.source_sha256,**affine.source_sha256,**(gate.source_sha256 if gate else {})}
  hashes[str(proof)]=hashlib.sha256(proof.read_bytes()).hexdigest()
  for m in [identity,fma_support,direct]:hashes[str(Path(m.__file__))]=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
  hashes[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA full original direct layer_norm + FMA'+(' + Mul' if zero else ''),'Frozen norm oracle is scale*LayerNorm(x)+shift, optional gate multiply, no implicit1+scale. Original explicit FP32 casts/output dtype/eps preserved; all complete normalization neutralweights/zeros/pack/pad/native/fma clone/copy/cast/fullgate counted. No kernel/block/flags/checkerchange.',hashes,cleanup=cleanup)
 return build
