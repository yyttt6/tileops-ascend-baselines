"""Audited CPU gold correction for exactly two BF16 BMM cases.
Applies to the case reference, independent of implementation/library.
Input/output dtypes and the original numerical checker are unchanged.
"""
import torch
SHAPES={
 ('BmmFwdOp','mha-decode-pv-bfloat16'):((64,128,2048),(64,2048,128)),
 ('BmmFwdOp','square-512-bfloat16'):((16,512,512),(16,512,512)),
}
def corrected_reference(op,case_id):
 key=(op,case_id)
 if key not in SHAPES:return None
 expected_shapes=SHAPES[key]
 def reference(a,b):
  assert a.device.type==b.device.type=='cpu','CPU oracle only; never a device implementation'
  assert a.dtype==b.dtype==torch.bfloat16
  assert (tuple(a.shape),tuple(b.shape))==expected_shapes
  # BF16 inputs widen exactly; accumulate in FP64, then round to the declared output dtype.
  return torch.bmm(a.to(torch.float64),b.to(torch.float64)).to(a.dtype)
 return reference
