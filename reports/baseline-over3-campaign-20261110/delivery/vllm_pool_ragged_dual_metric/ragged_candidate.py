"""Same-width ragged batching; every call uses current tensor metadata."""
def build(native,chunk,fixed):
 import torch
 assert fixed
 def run(x,offsets=None,indices=None):
  assert offsets is not None and indices is not None
  bounds=offsets.tolist()
  spans=[(bounds[s]+c*chunk,min(bounds[s]+(c+1)*chunk,bounds[s+1])) for s,c in indices.tolist()]
  groups={}
  for i,(begin,end) in enumerate(spans):
   assert end>begin
   groups.setdefault(end-begin,[]).append((i,begin,end))
  outputs=[None]*len(spans)
  for width,items in groups.items():
   packed=torch.stack([x[:,begin:end].float().contiguous() for _,begin,end in items],dim=1)
   means=native.mean_dim(packed,2,dtype=torch.float32).to(x.dtype)
   for j,(i,_,_) in enumerate(items):outputs[i]=means[:,j]
  return torch.stack(outputs,dim=1)
 return run
