from pathlib import Path
import os,sys,importlib.abc,json,types,hashlib,datetime
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P),str(P/'workspace/TileOPs/src'),str(P/'workspace/TileOPs')];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU proof blocks '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
import torch.nn.functional as F
from workloads.device import set_run_device
from ops_math_pool_indices_support import pool_indices_candidate
set_run_device('cpu');torch.set_num_threads(4);T=types.SimpleNamespace;checks=[]
for nd,shape,adaptive,dt in [(1,(2,3,13),False,torch.float16),(2,(1,2,7,9),False,torch.float32),(3,(1,2,5,7,9),False,torch.bfloat16),(2,(1,2,7,9),True,torch.float16)]:
 for ties in [False,True]:
  cpu=-(torch.arange(__import__('math').prod(shape)).reshape(shape)%193+1).to(dt)/8
  if ties:cpu.fill_(-2)
  opts={'output_size':(3,4)} if adaptive else dict(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=True)
  gold=getattr(F,'adaptive_max_pool2d' if adaptive else 'max_pool'+str(nd)+'d')(cpu,**opts,return_indices=True);params={**opts,'input':(shape,str(dt).split('.')[-1]),'output':(tuple(gold[0].shape),str(dt).split('.')[-1])};before=cpu.view(torch.uint8).clone()
  def child(c,op):
   assert op=='AmaxFwdOp' and c.params['_argvalue']
   def run(x):
    value,index=x.max(1);return index,value
   return T(implementation=T(run=run),cleanup=lambda:None)
  candidate=pool_indices_candidate(T(params=params),'AdaptiveMaxPool2dIndicesFwdOp' if adaptive else 'MaxPool'+str(nd)+'dIndicesFwdOp',child,lambda:{});out=candidate.implementation.run(cpu);assert torch.equal(out[0],gold[0]) and torch.equal(out[1],gold[1]) and torch.equal(cpu.view(torch.uint8),before);checks.append({'ndim':nd,'shape':shape,'dtype':str(dt),'ties':ties,'both_full_outputs_and_input_bits_exact':True})
proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_NPU_imported':True,'CPU_surrogate_only':True,'checks':checks,'sources':{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['ops_math_pool_indices_support.py','prove_math_pool_indices_cpu_v2.py','workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/pool_indices.py']}};(P/'MATH_POOL_INDICES_CPU_PROOF_V2.json').write_text(json.dumps(proof,indent=2)+'\n');print('POOL_INDICES_CPU_EXACT_TWOOUTPUT_TIES_PASS',len(checks))
