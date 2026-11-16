"""Layout-only packing for the original GeluMul width/batch contract."""
def gelu_mul(binding,x,attrs):
    import torch
    import torch.nn.functional as F
    width=x.shape[-1]//2;chunk=min(width,512);padding=(-width)%chunk
    left=F.pad(x[...,:width].reshape(-1,width),(0,padding)).contiguous().reshape(-1,chunk)
    right=F.pad(x[...,width:].reshape(-1,width),(0,padding)).contiguous().reshape(-1,chunk)
    packed=torch.cat((left,right),dim=-1);out=torch.empty_like(left)
    for start in range(0,packed.shape[0],200000):
        binding.call('GeluMul',[packed[start:start+200000]],[out[start:start+200000]],attrs)
    return out.reshape(-1,width+padding)[:,:width].contiguous().reshape(*x.shape[:-1],width)
