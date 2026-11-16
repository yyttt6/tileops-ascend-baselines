"""One native-layout alternative; complete SGL kernel unchanged."""
from sgl_layout_v2 import ROOT,factory as prior_factory
import math,sys

def factory(op,params,fixed):
 assert op=='MulFwdOp'
 if not fixed:return prior_factory(op,params,True)
 import torch
 sys.path.insert(0,str(ROOT))
 from sgl_kernel_npu.norm.scale_shift import fused_scale_shift
 def run(x,y):
  assert x.ndim==4 and y.ndim==3 and tuple(y.shape)==(x.shape[1],1,1)
  n,c,h,w=x.shape
  xx=x.contiguous().reshape(n,1,c*h*w)
  yy=y.expand(c,h,w).contiguous().reshape(-1)
  return fused_scale_shift(xx,yy,torch.zeros_like(xx),scale_constant=0.0).reshape(x.shape)
 return run
