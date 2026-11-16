"""Review-only proposal for ops-math/InstanceNorm/image-affine-float16 (ordinal309).
Not imported by any worker. Requires explicit case-specific user approval.
"""
def independent_reference_309(x,running_mean,running_var,weight,bias,*,eps=1e-5,momentum=0.1):
 import torch.nn.functional as F
 assert running_mean is None and running_var is None
 return F.instance_norm(x.double(),None,None,None if weight is None else weight.double(),None if bias is None else bias.double(),True,momentum,eps).to(x.dtype)
