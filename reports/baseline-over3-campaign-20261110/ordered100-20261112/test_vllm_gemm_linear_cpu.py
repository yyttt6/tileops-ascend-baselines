import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import pathlib,sys,ast,types,json,datetime,hashlib,itertools
import torch
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P/'workspace/isolated_wrappers'));import vllm_gemm_linear_api as helper
path=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/vllm/basic.py';mod=types.ModuleType('frozen_basic');mod.__file__=str(path);mod.Path=pathlib.Path;mod.validate=lambda case:None;mod.TRITON_ROOT=''
class Native:
 def matmul_persistent(self,a,b):return (a.float()@b.float()).to(a.dtype)
 def linear_persistent(self,a,w):return (a.float()@w.transpose(-1,-2).contiguous().float()).to(a.dtype)
mod.module=lambda path:(Native(),[]);mod.candidate=lambda run,*a,**kw:run
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='gemm');exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),vars(mod));alternate,proof=helper.clone_gemm(mod);cases=[]
for dtype,(m,k,n),ta,tb in itertools.product([torch.float16,torch.bfloat16,torch.float32],[(3,7,5),(16,32,24)],[False,True],[False,True]):
 a=torch.randn((k,m) if ta else (m,k),dtype=dtype);b=torch.randn((n,k) if tb else (k,n),dtype=dtype);case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='GemmFwdOp')),params={'trans_a':ta,'trans_b':tb})
 before=[x.clone() for x in [a,b]];old=mod.gemm(case);new=alternate(case);x,y=old(a,b),new(a,b);assert torch.equal(x,y) and all(torch.equal(t,u) for t,u in zip([a,b],before));cases.append({'dtype':str(dtype),'mnk':[m,n,k],'trans_a':ta,'trans_b':tb,'equal':True,'inputs_preserved':True})
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':len(cases),'rows':cases,'proof':proof,'helper_sha256':hashlib.sha256(pathlib.Path(helper.__file__).read_bytes()).hexdigest(),'CPU_surrogate_only_not_native_acceptance':True,'no_NPU_imported':not any(x.startswith('torch_npu') for x in sys.modules)};assert out['no_NPU_imported'];(P/'VLLM_GEMM_LINEAR_CPU_PROOF.json').write_text(json.dumps(out,indent=2));print('CPU_MAPPING_AND_AST_CASES',len(cases))
