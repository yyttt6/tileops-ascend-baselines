from pathlib import Path
import os,sys,importlib.abc,json,datetime,hashlib,types
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P),str(P/'workspace/TileOPs/src'),str(P/'workspace/TileOPs')];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU proof blocks '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
from workloads.device import set_run_device
from workloads.pool import AvgPoolWorkload
from workloads.numerics import reference_tolerance
from ops_math_weighted_pool_support import weighted_pool_candidate
T=types.SimpleNamespace;set_run_device('cpu');torch.set_num_threads(4);checks=[]
for nd,shape,dt in [(2,(1,2,7,9),torch.bfloat16),(2,(1,2,7,9),torch.float32),(3,(1,2,5,7,9),torch.bfloat16),(3,(1,2,5,7,9),torch.float32)]:
 opts=dict(kernel_size=3,stride=2,padding=1,ceil_mode=nd==2,count_include_pad=nd==3,divisor_override=7);x=(torch.arange(__import__('math').prod(shape)).reshape(shape)%97-48).to(dt)/16;before=x.view(torch.uint8).clone();w=AvgPoolWorkload.__new__(AvgPoolWorkload);w.ndim=nd;w.kernel_size=opts['kernel_size'];w.stride=opts['stride'];w.padding=opts['padding'];w.ceil_mode=opts['ceil_mode'];w.count_include_pad=opts['count_include_pad'];w.divisor_override=7;gold=w.ref_program(x)
 def child(c,op):
  assert op=='MeanFwdOp' and c.params['_weighted'];return T(implementation=T(run=lambda a,b,c:((a*b)/c).sum(-1)),cleanup=lambda:None)
 c=weighted_pool_candidate(T(params={**opts,'input':(shape,str(dt).split('.')[-1]),'output':(tuple(gold.shape),str(dt).split('.')[-1])}),'AvgPool'+str(nd)+'dFwdOp',child,lambda:{});out=c.implementation.run(x);torch.testing.assert_close(out,gold,**reference_tolerance(dt));assert torch.equal(x.view(torch.uint8),before);checks.append({'ndim':nd,'shape':shape,'dtype':str(dt),'original_reference_and_inputs_passed':True})
proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_NPU_imported':True,'CPU_surrogate_only':True,'checks':checks,'sources':{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['ops_math_weighted_pool_api.py','ops_math_weighted_pool_support.py','prove_math_weighted_pool_cpu.py']}};(P/'MATH_WEIGHTED_POOL_CPU_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n');print('CPU_WEIGHTED_POOL_PASS',len(checks))
