from pathlib import Path
import sys,os,importlib.abc,ast,json,hashlib,datetime,pickle
P=Path(__file__).resolve().parent;T=P/'workspace/TileOPs';sys.path[:0]=[str(T/'src'),str(T),str(P)];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU layout proof forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
from workloads.device import set_run_device
from workloads.numerics import compare_outputs
from grouped_scan_layout import construct,build
set_run_device('cpu');torch.set_num_threads(8);fn,proof=construct(torch);source=P/'workspace/TileOps-08f-original/workloads/mamba.py';node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='selective_scan_ref');env={'torch':torch,'F':torch.nn.functional};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),env);original=env['selective_scan_ref'];torch.manual_seed(0);tested=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for groups in [1,2,4]:
  B,D,N,L=2,8,4,5;u=torch.randn(B,D,L).to(dtype);dt=torch.randn_like(u);A=-torch.rand(D,N);b=torch.randn(B,groups,N,L).to(dtype);c=torch.randn_like(b);res=torch.randn(D);z=torch.randn_like(u);bias=torch.randn(D);seq=torch.tensor([[0,0,1,1,2],[0,1,1,2,2]],dtype=torch.int32);initial=torch.randn(B,D,N)
  for options in [False,True]:
   xs=(u,dt,A,b,c,res if options else None,z if options else None,bias if options else None,seq if options else None,initial if options else None);a=original(*xs,delta_softplus=options);v=fn(*xs,delta_softplus=options);assert all(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)) for x,y in zip(a,v,strict=True));tested.append({'dtype':str(dtype),'groups':groups,'all_optional_reset_and_initial':options,'both_outputs_bit_exact_vs_original_FP32':True})
d=P/'prepared-rollout/1498';case=pickle.load((d/'case.pkl').open('rb'));xs=torch.load(d/'payload.pt',weights_only=False)['inputs'];case.__dict__['inputs']=xs;measured,_=build(case);expected=case.reference(*xs);check=compare_outputs(measured(*xs),expected,case.verification);assert check.checked_outputs==check.total_outputs==2 and not check.unchecked_reason;assert not any(x.split('.')[0] in ('torch_npu','acl') for x in sys.modules);out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'proof':proof,'toy_cases':tested,'real1498_original_frozen_FP64_oracle_checker_both_outputs_passed':True,'no_NPU_imported':True,'zero_native_qualification':True,'helper_sha256':hashlib.sha256((P/'grouped_scan_layout.py').read_bytes()).hexdigest()};(P/'GROUPED_SCAN_LAYOUT_CPU_PROOF.json').write_text(json.dumps(out,indent=2));print('GROUPED_LAYOUT_CPU18_BITEXACT_AND_REAL_FROZEN_CHECKER_PASS',flush=True)
