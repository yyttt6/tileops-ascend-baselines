"""Three complete same-pin APIs matching original ReduceMeanWithCount DAG; no Torch math fallback."""
from pathlib import Path
import hashlib,json
from ops_math_dequant_api import MathDequantAPI
from ops_math_realdiv_capi_v2 import DivAPI
from ops_math_l2_api import SumOnly
class WeightedMeanAPI:
 def __init__(self,torch):
  self.t=torch;self.mul=MathDequantAPI(torch,'Mul');self.div=DivAPI(torch);self.sum=SumOnly(torch,dim=-1,outdtype=torch.float32);self.sources={**self.mul.sources,**self.sum.sources};f=self.div.N/'DIV_PACKAGE_V2.json';self.sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  for name,r in self.div.m['files'].items():self.sources[str(self.div.N/'graph-private-v2'/name)]=r['sha256']
  for name in ['ops_math_weighted_pool_api.py','ops_math_realdiv_capi_v2.py','ops_math_l2_api.py','ops_math_sqrt_api.py','ops_math_l1_api.py','ops_math_dequant_api.py']:
   f=Path(__file__).with_name(name);self.sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 def __call__(self,x,count,count_sum):
  assert x.ndim==2 and x.dtype==count.dtype==count_sum.dtype==self.t.float32 and x.shape==count.shape==count_sum.shape
  value=self.mul(x,count);value=self.div(value,count_sum);out=self.sum(value);self.cleanup();return out
 def cleanup(self):
  self.sum.cleanup();self.div.cleanup();self.mul.cleanup()
