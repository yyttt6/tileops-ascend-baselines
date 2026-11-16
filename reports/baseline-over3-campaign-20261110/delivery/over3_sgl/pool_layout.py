"""Batch independent uniform mean-pooling rows into unchanged complete SGL API."""
from .mul_layout import ROOT,factory as mul_factory
import sys

def factory(op,params,fixed):
 if op!='MeanPoolingFwdOp':return mul_factory(op,params,fixed)
 import torch
 sys.path.insert(0,str(ROOT))
 from sgl_kernel_npu.fla.layernorm_gated import layer_norm_fwd_npu_simd,layer_norm_fwd_npu
 native=layer_norm_fwd_npu if fixed else layer_norm_fwd_npu_simd
 chunk=params['chunk_size']
 def run(x,offsets=None,indices=None):
  if offsets is not None or indices is not None:raise NotImplementedError('Only uniform actual requests admitted')
  b,t,h,d=x.shape
  if t%chunk:raise NotImplementedError('Unpadded full chunks only')
  xx=x.reshape(b,t//chunk,chunk,h,d).permute(0,1,3,4,2).contiguous().reshape(-1,chunk)
  w=torch.ones(chunk,device=x.device,dtype=x.dtype)
  mean=native(xx,w,None,1e-5)[1]
  return mean.to(x.dtype).reshape(b,t//chunk,h,d)
 return run
