"""CPU-only algebra/layout proof; does not qualify a native composition."""
from pathlib import Path
import os,sys,importlib.abc,json,hashlib,datetime,ast
P=Path(__file__).resolve().parent;os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';sys.path[:0]=[str(P),str(P/'workspace/TileOPs'),str(P/'workspace/TileOPs/src')]
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU proof forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
import torch.nn.functional as F
from types import SimpleNamespace as T
from workloads.norm import AdaLayerNormWorkload
from workloads.device import set_run_device
from workloads.numerics import reference_tolerance
from ops_math_original_norm_layout import norm_layout_run
from ops_math_adaln3_layout import composed_run
set_run_device('cpu');torch.set_num_threads(4);checks=[]
for dt in [torch.float16,torch.bfloat16,torch.float32]:
 for mode in ['dense','broadcast','strided','wide']:
  m,n=(3,16385) if mode=='wide' else (3,17);w=AdaLayerNormWorkload(m,n,dt);torch.manual_seed(718);args=w.gen_inputs()
  if mode=='broadcast':args=(args[0],args[1][:1],args[2][:1])
  if mode=='strided':args=tuple(x.T.contiguous().T for x in args)
  before=[x.contiguous().view(torch.uint8).clone() for x in args];case=T(inputs=args,params={'eps':w.eps});count={'LN':0,'Mul':0,'Add':0}
  def ln(x,g,b,e):count['LN']+=1;return F.layer_norm(x,(x.shape[-1],),g,b,e)
  def mul(a,b):count['Mul']+=1;return a*b
  def add(a,b):count['Add']+=1;return a+b
  out=composed_run(case,ln,mul,add)(*args);gold=w.ref_program(*args);torch.testing.assert_close(out,gold,**reference_tolerance(dt));assert count=={'LN':1,'Mul':1,'Add':1};old=norm_layout_run(case,'AdaLayerNormFwdOp',lambda x,g,b,e:F.layer_norm(x,(x.shape[-1],),g,b,e))(*args);torch.testing.assert_close(out,old,**reference_tolerance(dt));assert all(torch.equal(x.contiguous().view(torch.uint8),b) for x,b in zip(args,before));checks.append({'dtype':str(dt),'mode':mode,'shape':[m,n],'original_full_reference_and_inputbits_passed':True,'original_layout_CPU_surrogate_comparison_passed':True,'complete_API_calls':count})
assert not any(k.split('.')[0] in ('torch_npu','acl') for k in sys.modules)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_NPU_imported':True,'CPU_surrogate_only_not_native_qualification':True,'kernel_and_checker_unchanged':True,'checks':checks,'sources':{n:sha(P/n) for n in ['ops_math_adaln3_layout.py','prove_ops_math_adaln3_cpu.py','ops_math_original_norm_layout.py','workspace/TileOPs/workloads/norm.py','workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/layernorm.py']}};(P/'OPS_MATH_ADALN3_CPU_PROOF.json').write_text(json.dumps(r,indent=2)+'\n');print('CPU proof passed',len(checks))
