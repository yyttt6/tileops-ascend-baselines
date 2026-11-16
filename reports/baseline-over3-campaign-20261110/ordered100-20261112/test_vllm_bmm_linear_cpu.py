import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import pathlib,sys,ast,types,json,datetime,hashlib,itertools
import torch
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P/'workspace/isolated_wrappers'));import vllm_bmm_linear_api as helper
path=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/vllm/basic.py';mod=types.ModuleType('frozen_basic');mod.__file__=str(path);mod.Path=pathlib.Path;mod.validate=lambda c:None;mod.TRITON_ROOT='';calls=[]
class Native:
 def matmul_persistent(self,a,b):calls.append('matmul');return (a.float()@b.float()).to(a.dtype)
 def linear_persistent(self,a,w):calls.append('linear');return (a.float()@w.transpose(-1,-2).contiguous().float()).to(a.dtype)
mod.module=lambda path:(Native(),[]);mod.candidate=lambda run,*a,**kw:run
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='gemm');exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),vars(mod));alternate,proof=helper.clone_bmm(mod);rows=[]
for dtype,(batch,m,k,n),strided in itertools.product([torch.float16,torch.bfloat16,torch.float32],[(2,3,7,5),(3,16,32,24)],[False,True]):
 a=torch.randn((batch,m,k*(2 if strided else 1)),dtype=dtype);b=torch.randn((batch,k,n*(2 if strided else 1)),dtype=dtype)
 if strided:a=a[...,::2];b=b[...,::2]
 before=[x.clone() for x in [a,b]];case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='BmmFwdOp')),params={});calls.clear();old=mod.gemm(case)(a,b);assert calls==['matmul']*batch;calls.clear();new=alternate(case)(a,b);assert calls==['linear']*batch and torch.equal(old,new) and all(torch.equal(x,y) for x,y in zip([a,b],before));rows.append({'dtype':str(dtype),'batch_mkn':[batch,m,k,n],'strided':strided,'bitwise_equal':True,'all_batches_executed':True,'inputs_preserved':True})
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':len(rows),'rows':rows,'proof':proof,'helper_sha256':hashlib.sha256(pathlib.Path(helper.__file__).read_bytes()).hexdigest(),'CPU_surrogate_only_not_native_acceptance':True,'no_NPU_imported':not any(k.startswith('torch_npu') for k in sys.modules)};assert out['no_NPU_imported'];(P/'VLLM_BMM_LINEAR_CPU_PROOF.json').write_text(json.dumps(out,indent=2));print('CPU_BATCHED_MAPPING_CASES',len(rows))
