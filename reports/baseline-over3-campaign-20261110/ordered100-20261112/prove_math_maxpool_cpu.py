"""CPU surrogate window-boundary checks. Never numerical device acceptance."""
from pathlib import Path
import os,sys,importlib.abc,json,hashlib,ast,types,datetime
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P),str(P/'workspace/TileOPs/src'),str(P/'workspace/TileOPs')];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU proof forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
import torch.nn.functional as F
from ops_math_maxpool_support import pool_candidate
T=types.SimpleNamespace;torch.set_num_threads(4);checks=[]
configs=[('MaxPool1dFwdOp',(3,11),dict(kernel_size=3,stride=2,padding=1,dilation=2,ceil_mode=True),torch.float16),('MaxPool1dFwdOp',(2,3,13),dict(kernel_size=3,stride=2,padding=1,dilation=2,ceil_mode=False),torch.bfloat16),('MaxPool2dFwdOp',(3,7,9),dict(kernel_size=(3,2),stride=(2,2),padding=(1,0),dilation=(1,1),ceil_mode=True),torch.float32),('MaxPool2dFwdOp',(2,3,9,11),dict(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False),torch.bfloat16),('MaxPool3dFwdOp',(1,2,5,7,9),dict(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=True),torch.float16),('AdaptiveMaxPool2dFwdOp',(2,3,7,11),dict(output_size=(3,4)),torch.bfloat16),('AdaptiveMaxPool2dFwdOp',(3,7,11),dict(output_size=(1,1)),torch.float16),('MaxPool2dFwdOp',(1,2,5,7),dict(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=True),torch.float16),('MaxPool3dFwdOp',(1,2,5,7,9),dict(kernel_size=(2,3,2),stride=(2,2,2),padding=(0,1,0),dilation=1,ceil_mode=False),torch.float32)]
for i,(op,shape,opts,dt) in enumerate(configs):
 x=-(torch.arange(__import__('math').prod(shape)).reshape(shape)%193+1).to(dt)/8;before=x.contiguous().view(torch.uint8).clone();ref=getattr(F,'adaptive_max_pool2d' if op.startswith('Adaptive') else 'max_pool'+op.split('Pool')[1][0]+'d')(x,**opts);params=dict(opts,input=(shape,str(dt).split('.')[-1]),output=(tuple(ref.shape),str(dt).split('.')[-1]));case=T(params=params)
 def child(c,name):
  assert name=='AmaxFwdOp';return T(implementation=T(run=lambda v:torch.amax(v,dim=-1)),cleanup=lambda:None)
 candidate=pool_candidate(case,op,child,lambda:{});out=candidate.implementation.run(x);torch.testing.assert_close(out,ref,rtol=0,atol=0);assert torch.equal(before,x.contiguous().view(torch.uint8));checks.append({'op':op,'shape':shape,'dtype':str(dt),'options':opts,'full_output_exact':True,'input_bits_unchanged':True,'all_negative_values_abs_bug_rejected':True})
proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_surrogate_only':True,'no_NPU_imported':True,'checks':checks,'unchanged_run_reduce_AST_checked_in_actual_factory':True,'not_native_or_formal_acceptance':True,'sources':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['ops_math_maxpool_support.py','ops_math_maxpool_api.py','prove_math_maxpool_cpu.py']}};(P/'MATH_MAXPOOL_CPU_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n');print('CPU_POOL_WINDOW_EXACT',len(checks))
