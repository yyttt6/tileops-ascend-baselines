"""AdaLN complete LayerNorm+Mul+Add composition. No device arithmetic fallback."""
def composed_run(case,native,mul,add):
 import torch
 eps=case.params.get('eps');eps=torch.finfo(torch.float32).eps if eps is None else eps
 def run(x,scale,shift):
  shape=x.shape;k=shape[-1];xx=x.float().contiguous().reshape(-1,k)
  gamma=torch.ones((k,),dtype=torch.float32,device=x.device);beta=torch.zeros((k,),dtype=torch.float32,device=x.device)
  normed=native(xx,gamma,beta,eps).reshape(shape)
  sc=scale.float().expand(shape).contiguous();sh=shift.float().expand(shape).contiguous()
  return add(mul(normed,sc),sh).to(x.dtype)
 return run
