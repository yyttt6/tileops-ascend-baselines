"""Avoid identity padding in SGL complete add/sub APIs; native kernels unchanged."""
from .mul_layout import ROOT,factory as prior_factory
import sys,math

def factory(op,params,fixed):
 if op not in ('AddFwdOp','SubFwdOp'):return prior_factory(op,params,fixed)
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(ROOT))
 from sgl_kernel_npu.moe.mul_add import mul_add
 alpha=params.get('alpha',1)
 def run(x,y):
  shape=torch.broadcast_shapes(x.shape,y.shape);size=math.prod(shape);tail=-size%4096
  def pack(t):
   t=t.expand(shape).contiguous().reshape(-1)
   if not fixed or tail:t=F.pad(t,(0,tail))
   return t.reshape(-1,4096)
  return mul_add(pack(y),pack(x),alpha if op=='AddFwdOp' else -alpha).reshape(-1)[:size].reshape(shape)
 return run
