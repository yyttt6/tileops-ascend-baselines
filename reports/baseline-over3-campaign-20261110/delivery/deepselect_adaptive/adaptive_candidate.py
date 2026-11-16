"""Batch same-width windows through original TopK; no kernel/math changes."""
import math,collections

def factory(module,semantics,params):
 import torch
 shape,out_spatial,windows=semantics.adaptive_windows(params)
 planes=math.prod(shape[:-2]);align=module.get_stride_requirement()[0]//2
 groups=collections.defaultdict(list)
 for i,(hs,he,ws,we) in enumerate(windows):groups[(he-hs)*(we-ws)].append((i,hs,he,ws,we))
 order=[i for items in groups.values() for i,*_ in items];inverse=[order.index(i) for i in range(len(windows))]
 def run(x):
  view=x.reshape(planes,*shape[-2:]);outputs=[]
  for width,items in groups.items():
   packed=torch.stack([view[:,hs:he,ws:we].reshape(planes,width) for _,hs,he,ws,we in items],dim=1).reshape(-1,width)
   values=[]
   for start in range(0,packed.shape[0],32768):
    piece=packed[start:start+32768];buffer=torch.empty((piece.shape[0],(width+align-1)//align*align),dtype=x.dtype,device=x.device)
    buffer[:,:width].copy_(piece)
    v,_=module.topk(buffer[:,:width].view(torch.bfloat16),1,sorted=False,indices_type=torch.int32,return_value=True,abort_when_nan_found=True)
    values.append(v.contiguous().view(x.dtype).reshape(-1))
   outputs.append(torch.cat(values).reshape(planes,len(items)))
  joined=torch.cat(outputs,dim=1)
  restore=torch.tensor(inverse,dtype=torch.int64,device=x.device)
  return joined.index_select(1,restore).reshape(*shape[:-2],*out_spatial)
 return run
