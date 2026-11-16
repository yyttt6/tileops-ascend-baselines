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
from workloads.numerics import reference_tolerance
from ops_math_pool_mean_support import mean_pool_candidate
T=types.SimpleNamespace;torch.set_num_threads(4);checks=[]
configs=[('AvgPool1dFwdOp',(3,11),dict(kernel_size=3,stride=2,padding=1,ceil_mode=True,count_include_pad=False),torch.float16),('AvgPool1dFwdOp',(2,3,13),dict(kernel_size=3,stride=2,padding=1,ceil_mode=True,count_include_pad=True),torch.bfloat16),('AvgPool2dFwdOp',(3,7,9),dict(kernel_size=(3,2),stride=(2,2),padding=(1,0),ceil_mode=True,count_include_pad=True),torch.float32),('AvgPool2dFwdOp',(2,3,9,11),dict(kernel_size=3,stride=2,padding=1,ceil_mode=False,count_include_pad=False),torch.bfloat16),('AvgPool3dFwdOp',(1,2,5,7,9),dict(kernel_size=2,stride=2,padding=0,ceil_mode=True,count_include_pad=True),torch.float16),('AdaptiveAvgPool2dFwdOp',(2,3,7,11),dict(output_size=(3,4)),torch.bfloat16),('AdaptiveAvgPool2dFwdOp',(3,7,11),dict(output_size=(1,1)),torch.float16),('AvgPool2dFwdOp',(1,2,5,7),dict(kernel_size=3,stride=2,padding=1,ceil_mode=True,count_include_pad=False),torch.float16),('AvgPool3dFwdOp',(1,2,5,7,9),dict(kernel_size=(2,3,2),stride=(2,2,2),padding=(0,1,0),ceil_mode=False,count_include_pad=False),torch.float32)]
for i,(op,shape,opts,dt) in enumerate(configs):
 x=-(torch.arange(__import__('math').prod(shape)).reshape(shape)%193+1).to(dt)/8;before=x.contiguous().view(torch.uint8).clone();ref=getattr(F,'adaptive_avg_pool2d' if op.startswith('Adaptive') else 'avg_pool'+op.split('Pool')[1][0]+'d')(x,**opts);params=dict(opts,input=(shape,str(dt).split('.')[-1]),output=(tuple(ref.shape),str(dt).split('.')[-1]));case=T(params=params)
 def child(c,name):
  assert name=='MeanFwdOp';return T(implementation=T(run=lambda v:torch.mean(v,dim=-1)),cleanup=lambda:None)
 candidate=mean_pool_candidate(case,op,child,lambda:{});out=candidate.implementation.run(x);torch.testing.assert_close(out,ref,**reference_tolerance(ref.dtype));assert torch.equal(before,x.contiguous().view(torch.uint8));checks.append({'op':op,'shape':shape,'dtype':str(dt),'options':opts,'full_output_original_dtype_tolerance_passed':True,'tolerance':reference_tolerance(ref.dtype),'input_bits_unchanged':True,'count_include_pad_ceil_adaptive_checked':True})
proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_surrogate_only':True,'no_NPU_imported':True,'checks':checks,'unchanged_run_reduce_AST_checked_in_actual_factory':True,'not_native_or_formal_acceptance':True,'sources':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['ops_math_pool_mean_support.py','ops_math_pool_mean_api.py','prove_math_pool_mean_cpu.py']}};(P/'MATH_POOL_MEAN_CPU_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n');print('CPU_MEAN_POOL_ORIGINAL_TOLERANCE',len(checks))
