"""Complete unchanged Math LayerNorm/Mul/Add APIs; optional final Mul before cast."""
from pathlib import Path
import dataclasses,hashlib
from ops_math_adaln3_support import make_candidate
from ops_math_dequant_api import MathDequantAPI
def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name in ('AdaLayerNormFwdOp','AdaLayerNormZeroFwdOp')
  import torch
  from benchmarks.api import Implementation
  base=make_candidate(case);zero=case._call.signature.name=='AdaLayerNormZeroFwdOp';gate=MathDequantAPI(torch,'Mul') if zero else None
  def run(x,scale,shift,*gating):
   assert len(gating)==int(zero)
   # Passing original FP32 widened x preserves the original reference's single
   # final dtype cast, including after the optional gate multiply.
   out=base.implementation.run(x.float(),scale,shift)
   if zero:out=gate(out,gating[0].float())
   return out.to(x.dtype)
  def cleanup():
   base.cleanup()
   if gate is not None:gate.cleanup()
  sources=dict(base.source_sha256)
  if gate is not None:sources.update(gate.sources)
  f=Path(__file__);sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return dataclasses.replace(base,implementation=Implementation(run),api='ops-math original complete LayerNorm + Mul + Add'+(' + final Mul' if zero else ''),adaptation='Original full FP32 normalization/affine/optional gate and one final storage-dtype cast. All native calls, neutral affine allocation, packing, casts and full outputs timed. No kernel/compiler/checker/tolerance/input changes.',source_sha256=sources,cleanup=cleanup)
 return build
