def build(select, affine_api):
 import torch
 import torch.nn.functional as F
 def affine(x,y,factor):
  shape=torch.broadcast_shapes(x.shape,y.shape)
  xx=x.expand(shape).contiguous().reshape(-1)
  yy=y.expand(shape).contiguous().reshape(-1)
  size=xx.numel();tail=-size%4096
  return affine_api(F.pad(xx,(0,tail)).reshape(-1,4096),F.pad(yy,(0,tail)).reshape(-1,4096),factor).reshape(-1)[:size].reshape(shape)
 def ne(x):
  y=torch.zeros((),device=x.device,dtype=x.dtype)
  shape=torch.broadcast_shapes(x.shape,y.shape)
  xx,yy=x.expand(shape),y.expand(shape)
  def less(a,b):
   return select(torch.stack((a,b),dim=-1).reshape(-1,2))[0].float().reshape(shape)
  return affine(less(xx,yy),less(yy,xx),1.0).to(torch.bool)
 return ne
