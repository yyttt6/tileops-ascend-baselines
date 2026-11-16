"""Same unmodified SGL scale_shift API; broadcast-only layout correction."""
import pathlib,sys,hashlib,math
ROOT=pathlib.Path('/opt/baseline-over3-20261110/sgl-pool-api-server1/sgl-source')
def factory(op,params,fixed):
 if op!='MulFwdOp':raise NotImplementedError(op)
 import torch
 sys.path.insert(0,str(ROOT))
 from sgl_kernel_npu.norm.scale_shift import fused_scale_shift
 def run(x,y):
  shape=torch.broadcast_shapes(x.shape,y.shape)
  if not fixed:
   xx=x.expand(shape).contiguous().reshape(1,1,-1);yy=y.expand(shape).contiguous().reshape(-1)
   return fused_scale_shift(xx,yy,torch.zeros_like(xx),scale_constant=0.0).reshape(shape)
  ys=(1,)*(len(shape)-y.ndim)+tuple(y.shape)
  shared=[i for i in range(len(shape)) if ys[i]==1]
  varied=[i for i in range(len(shape)) if ys[i]!=1]
  if not shared:raise NotImplementedError('No broadcast axis; no shared scale layout benefit')
  order=shared+varied;inverse=[order.index(i) for i in range(len(shape))]
  ordered=tuple(shape[i] for i in order);width=math.prod(shape[i] for i in varied)
  xx=x.expand(shape).permute(order).contiguous().reshape(1,-1,width)
  yy=y.reshape(ys).permute(order).contiguous().reshape(-1)
  assert yy.numel()==width
  out=fused_scale_shift(xx,yy,torch.zeros_like(xx),scale_constant=0.0)
  return out.reshape(ordered).permute(inverse).contiguous()
 return run
