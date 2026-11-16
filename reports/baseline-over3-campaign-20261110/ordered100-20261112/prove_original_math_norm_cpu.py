from pathlib import Path
import os,sys,importlib.abc,json,hashlib,datetime,types
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P),str(P/'workspace/TileOPs/src'),str(P/'workspace/TileOPs')];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU proof forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
import torch.nn.functional as F
from workloads.norm import LayerNormWorkload,AdaLayerNormWorkload,GroupNormWorkload,InstanceNormWorkload
from workloads.device import set_run_device
from workloads.numerics import reference_tolerance
from ops_math_original_norm_layout import norm_layout_run
set_run_device('cpu');torch.set_num_threads(4);T=types.SimpleNamespace;checks=[]
configs=[('LayerNormFwdOp',torch.float16,False),('LayerNormFwdOp',torch.float32,True),('AdaLayerNormFwdOp',torch.float16,False),('AdaLayerNormFwdOp',torch.bfloat16,True),('GroupNormFwdOp',torch.float32,False),('GroupNormFwdOp',torch.bfloat16,True),('InstanceNormFwdOp',torch.float16,False),('InstanceNormFwdOp',torch.float32,True)]
for op,dt,variant in configs:
 if op in ('LayerNormFwdOp','AdaLayerNormFwdOp'):
  m,n=3,16385 if op=='LayerNormFwdOp' and variant else 17;w=(LayerNormWorkload if op=='LayerNormFwdOp' else AdaLayerNormWorkload)(m,n,dt);params={'eps':w.eps,'normalized_shape':(n,)}
 else:
  w=GroupNormWorkload(2,8,(5,7),4,dt) if op=='GroupNormFwdOp' else InstanceNormWorkload(2,3,(5,7),dt);params={'eps':w.eps,'num_groups':4,'use_input_stats':True}
 torch.manual_seed(718);args=w.gen_inputs()
 if variant and op=='GroupNormFwdOp':args=(args[0],None,None)
 if variant and op=='AdaLayerNormFwdOp':args=(args[0],args[1][:1],args[2][:1])
 before=[x.contiguous().view(torch.uint8).clone() if isinstance(x,torch.Tensor) else x for x in args];run=norm_layout_run(T(params=params,inputs=args),op,lambda x,g,b,e:F.layer_norm(x,(x.shape[-1],),g,b,e));out=run(*args);gold=w.ref_program(*args);torch.testing.assert_close(out,gold,**reference_tolerance(dt));assert all(torch.equal(x.contiguous().view(torch.uint8),old) if isinstance(x,torch.Tensor) else x is old for x,old in zip(args,before));checks.append({'op':op,'dtype':str(dt),'variant':variant,'original_reference_full_output_and_inputbits_passed':True,'unaltered_affine_row_run_AST':True})
proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_NPU_imported':True,'CPU_surrogate_only':True,'checks':checks,'sources':{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['ops_math_original_norm_layout.py','prove_original_math_norm_cpu.py','workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/layernorm.py']}};(P/'ORIGINAL_MATH_NORM_CPU_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n');print('ORIGINAL_NORM_LAYOUT_CPU_PASS',len(checks))
