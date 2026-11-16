"""PROPOSAL ONLY. Not deployed/activated; needs explicit userapproval for these3 keys."""
KEYS={('ops-nn','GemmFwdOp','square-1k-bfloat16'),('ops-nn','GemmFwdOp','ds-v3-m64-down-bfloat16'),('ops-nn','GemmFwdOp','ds-v3-decode-down-bfloat16')}
def proposed_CPU_reference(torch,case):
 assert ('ops-nn',case._call.signature.name,case.id) in KEYS
 p=case.params;dtype=getattr(torch,p['d'][1]);assert dtype==torch.bfloat16
 def reference(a,b):
  assert a.device.type==b.device.type=='cpu' and a.dtype==b.dtype==dtype
  aa=a.T if p['trans_a'] else a;bb=b.T if p['trans_b'] else b
  return (aa.double()@bb.double()).to(dtype)
 return reference
