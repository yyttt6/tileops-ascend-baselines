from pathlib import Path
import sys,os,json,pickle,hashlib,ast,datetime,importlib.abc
P=Path(__file__).resolve().parent;T=P/'workspace/TileOps-08f-original';sys.path[:0]=[str(T/'src'),str(T),str(P)];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU preparation forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
from benchmarks.api import cases
from workloads.device import set_run_device
from workloads.numerics import compare_outputs
from selective_scan_bare_frozen_policy import bind
set_run_device('cpu');torch.set_num_threads(8);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();i=1499;task=json.loads((P/'TASKS.json').read_text())[i];published=json.loads((P/'SELECTIVE_SCAN_BARE_PUBLISHED_ROW.json').read_text());assert task['op']==published['op']=='SelectiveScanFwdOp' and task['case']==published['case']=='bare-scan-bfloat16';case=next(x for x in cases(task['op']) if x.id==task['case'])
def norm(x):
 if isinstance(x,torch.dtype):return str(x).removeprefix('torch.')
 if isinstance(x,(tuple,list)):return [norm(v) for v in x]
 if isinstance(x,dict):return {k:norm(v) for k,v in x.items()}
 return x
assert norm(case.params)==published['workload']['params'];d=P/'prepared-selective-scan-bare-08f'/str(i);d.mkdir(parents=True,exist_ok=True);assert not (d/'prepare.json').exists();torch.manual_seed(0)
with (d/'case.pkl').open('wb') as out:pickle.dump(case,out,protocol=5)
xs=case.inputs;before=[(x,x.contiguous().view(torch.uint8).clone()) for x in xs if isinstance(x,torch.Tensor)];reference,checker,policy=bind(case);expected=reference(*xs);source=T/'workloads/mamba.py';node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='selective_scan_ref');env={'torch':torch,'F':torch.nn.functional};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),env);produced=env['selective_scan_ref'](*xs,delta_softplus=case.params['delta_softplus']);check=compare_outputs(produced,expected,checker);assert check.checked_outputs==check.total_outputs==2 and not check.unchecked_reason;assert all(torch.equal(x.contiguous().view(torch.uint8),b) for x,b in before);assert not any(x.split('.')[0] in ('torch_npu','acl') for x in sys.modules);torch.save({'inputs':xs,'expected':expected},d/'payload.pt');sources={str(f):sha(f) for f in T.rglob('*') if f.is_file() and f.suffix in {'.py','.json','.yaml','.yml','.toml'} and '__pycache__' not in f.parts}
for file in [Path(__file__),P/'selective_scan_bare_frozen_policy.py',Path(policy['frozen_source']),Path(policy['frozen_numerics_source']),P/'TASKS.json',P/'SELECTIVE_SCAN_BARE_PUBLISHED_ROW.json']:sources[str(file)]=sha(file)
meta={'ordinal':i,'key':[task['library'],task['op'],task['case']],'source_commit':'08f66b153c44b47330e7fe4311470152f287a112','torch':torch.__version__,'params':norm(case.params),'seed':0,'reference_policy':'original_npu','frozen_oracle_and_checker':policy,'CPU_published_FP32_program_vs_original_frozen_FP64_oracle_and_checker_passed':True,'both_y_and_final_states_checked':True,'all_inputs_bit_preserved':True,'no_NPU_imported':True,'payload_sha256':sha(d/'payload.pt'),'descriptor_sha256':sha(d/'case.pkl'),'sources':sources};(d/'prepare.json').write_text(json.dumps(meta,indent=2));(P/'SELECTIVE_SCAN_BARE_ORIGINAL_CPU_PREPARATION.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_NPU_imported':True,'ordinal':i,'source_commit':meta['source_commit'],'prepare_sha256':sha(d/'prepare.json'),'payload_sha256':meta['payload_sha256'],'zero_native_qualification':True,'frozen_reference_checker_preserved':True},indent=2));print('CPU_BARE_SCAN_ORIGINAL_PARAMS_INPUTBITS_FROZEN_CHECKER_PASS',i,flush=True)
