"""Complete aclnnMean with explicit original FP32 accumulation and original output dtype."""
import ctypes as C
from ops_math_pool_mean_api import MeanAPI
class ChunkMeanAPI(MeanAPI):
 def __init__(self,torch,shape):
  super().__init__(torch,dim=-1,shape=shape,outdtype=torch.float32);self.codes[torch.float16]=1
 def __call__(self,x):
  assert x.ndim==5 and x.dtype in (self.t.float16,self.t.float32)
  r={'keep':[],'tensors':[],'arrays':[]};self.pending.append(r);x=x.contiguous();dims=(C.c_int64*1)(2);array=self.base.aclCreateIntArray(dims,1);assert array;r['keep'].append(dims);r['arrays'].append(array);shape=tuple(s for i,s in enumerate(x.shape) if i!=2);assert shape==self.shape;out=self.t.empty(shape,dtype=self.t.float32,device=x.device)
  self.call(self.mean_get,self.mean_run,[self.desc(x,r),array,False,0,self.desc(out,r)],r,x.device)
  result=out.to(x.dtype);r['keep'].append(result);return result
