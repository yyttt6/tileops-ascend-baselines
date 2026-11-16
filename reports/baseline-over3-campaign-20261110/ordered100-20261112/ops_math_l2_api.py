"""Three complete same-pin math APIs. No Torch arithmetic fallback."""
from pathlib import Path
import ctypes as C,hashlib,json
from ops_math_l1_api import MathL1API
from ops_math_dequant_api import MathDequantAPI
import ops_math_sqrt_api
class SumOnly(MathL1API):
 def __call__(self,x):
  assert x.ndim==2 and x.dtype==self.t.float32
  r={'keep':[],'tensors':[],'arrays':[]};self.pending.append(r);x=x.contiguous();dims=(C.c_int64*1)(1);array=self.base.aclCreateIntArray(dims,1);assert array;r['keep'].append(dims);r['arrays'].append(array);out=self.t.empty((x.shape[0],),dtype=self.t.float32,device=x.device)
  self.call(self.sumget,self.sum,[self.desc(x,r),array,False,self.codes[out.dtype],self.desc(out,r)],r,x.device);return out
class MathL2API:
 def __init__(self,torch):
  self.t=torch;self.mul=MathDequantAPI(torch,'Mul');self.sum=SumOnly(torch,dim=-1,outdtype=torch.float32);self.sqrt=ops_math_sqrt_api.MathSqrtAPI(torch,'Sqrt');self.sources={**self.mul.sources,**self.sum.sources,**self.sqrt.sources}
  for name in ['ops_math_l2_api.py','ops_math_sqrt_api.py','ops_math_l1_api.py','ops_math_dequant_api.py']:
   p=Path(__file__).with_name(name);self.sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 def __call__(self,packed,zero):
  assert packed.ndim==2 and packed.dtype==self.t.float32 and zero.dtype==self.t.float32 and tuple(zero.shape)==(1,packed.shape[1])
  squared=self.mul(packed,packed);reduced=self.sum(squared);out=self.sqrt(reduced);self.cleanup();return out.reshape(packed.shape[0],1)
 def cleanup(self):
  self.sqrt.cleanup();self.sum.cleanup();self.mul.cleanup()
