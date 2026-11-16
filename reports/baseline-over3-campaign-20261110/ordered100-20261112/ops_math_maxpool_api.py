"""Complete original Amax C API, without absolute-value preprocessing."""
import ctypes as C
from ops_math_vectorinf_api import VectorInfAPI
class MaxOnlyAPI(VectorInfAPI):
 def __call__(self,x):
  assert x.ndim==2 and x.dtype in (self.t.float16,self.t.bfloat16,self.t.float32)
  r={'keep':[],'tensors':[],'arrays':[]};self.pending.append(r);x=x.contiguous();dims=(C.c_int64*1)(1);array=self.base.aclCreateIntArray(dims,1);assert array;r['keep'].append(dims);r['arrays'].append(array);out=self.t.empty((x.shape[0],),dtype=x.dtype,device=x.device)
  assert self.shape is None or tuple(out.shape)==self.shape
  self.call(self.amaxget,self.amax,[self.desc(x,r),array,False,self.desc(out,r)],r,x.device)
  return out
