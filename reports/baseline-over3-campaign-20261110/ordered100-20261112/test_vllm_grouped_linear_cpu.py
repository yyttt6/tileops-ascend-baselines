import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import pathlib,sys,ast,types,json,datetime,hashlib,itertools
import torch
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P/'workspace/isolated_wrappers'));import vllm_grouped_linear_api as helper
path=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/vllm/basic.py';mod=types.ModuleType('frozen_basic');mod.__file__=str(path);mod.Path=pathlib.Path;mod.validate=lambda c:None;mod.TRITON_ROOT='';calls=[]
class Native:
 def matmul_persistent(self,a,b):calls.append('matmul');return (a.float()@b.float()).to(a.dtype)
 def linear_persistent(self,a,w):calls.append('linear');return (a.float()@w.transpose(-1,-2).contiguous().float()).to(a.dtype)
mod.module=lambda path:(Native(),[]);mod.candidate=lambda run,*a,**kw:run
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='grouped_gemm');exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),vars(mod));alternate,proof=helper.clone_grouped(mod);rows=[]
for dtype,ta,tb,strided in itertools.product([torch.float16,torch.bfloat16,torch.float32],[False,True],[False,True],[False,True]):
 k,n=7,5;plans=([2,3],[3,2]) if ta else ([2,0,3],[3,0,2]);a=torch.randn((5,k*(2 if strided else 1)),dtype=dtype)
 if strided:a=a[...,::2]
 if ta:b=torch.randn((n,5) if tb else (5,n),dtype=dtype)
 else:b=torch.randn((3,n,k) if tb else (3,k,n),dtype=dtype)
 case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='GroupedGemmFwdOp')),params={'transpose_a':ta,'transpose_b':tb});old=mod.grouped_gemm(case);new=alternate(case)
 for plan in plans:
  sizes=torch.tensor(plan,dtype=torch.int64);offsets=torch.tensor([sum(plan[:i]) for i in range(len(plan))],dtype=torch.int64);inputs=[a,b,sizes,offsets];before=[x.clone() for x in inputs];expected=len(plan) if ta else sum(x>0 for x in plan);calls.clear();x=old(*inputs);assert calls==['matmul']*expected;calls.clear();y=new(*inputs);assert calls==['linear']*expected and torch.equal(x,y) and all(torch.equal(t,u) for t,u in zip(inputs,before));rows.append({'dtype':str(dtype),'transpose_a':ta,'transpose_b':tb,'strided_a':strided,'sizes':plan,'bitwise_equal':True,'inputs_and_metadata_preserved':True,'all_nonempty_groups_executed':True})
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':len(rows),'rows':rows,'proof':proof,'dynamic_metadata_two_plans_per_factory':True,'helper_sha256':hashlib.sha256(pathlib.Path(helper.__file__).read_bytes()).hexdigest(),'CPU_surrogate_only_not_native_acceptance':True,'no_NPU_imported':not any(x.startswith('torch_npu') for x in sys.modules)};assert out['no_NPU_imported'];(P/'VLLM_GROUPED_LINEAR_CPU_PROOF.json').write_text(json.dumps(out,indent=2));print('CPU_GROUPED_MAPPING_CASES',len(rows))
