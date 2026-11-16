from pathlib import Path
import sys,os,json,pickle,hashlib,datetime,importlib.abc,shutil
P=Path(__file__).resolve().parent;T=P/'workspace/TileOps-08f-original';sys.path[:0]=[str(T/'src'),str(T),str(P)];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU preparation forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
from benchmarks.api import cases
from workloads.device import set_run_device
from selective_scan_cached_fidelity_expansion import validate_source
from selective_scan_denominator_frozen_policy import bind
set_run_device('cpu');torch.set_num_threads(8);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();tasks=json.loads((P/'TASKS.json').read_text());catalog={c.id:c for c in cases('SelectiveScanFwdOp')};accepted=[]
def norm(x):
 if isinstance(x,torch.dtype):return str(x).removeprefix('torch.')
 if isinstance(x,(tuple,list)):return [norm(v) for v in x]
 if isinstance(x,dict):return {k:norm(v) for k,v in x.items()}
 return x
for i in [1498,1497,1500,1501,1502]:
 old=P/'prepared-rollout'/str(i);m=json.loads((old/'prepare.json').read_text());sourceproof=validate_source(P,i,m);assert sha(old/'payload.pt')==m['payload_sha256'] and sha(old/'case.pkl')==m['descriptor_sha256'];case=catalog[tasks[i]['case']];saved=torch.load(old/'payload.pt',weights_only=False);xs=saved['inputs'];case.__dict__['inputs']=xs
 assert norm(case.params)==m['params'],'Frozen params differ from exact published case; stop before device'
 d=P/'prepared-selective-scan-denominator-08f'/str(i);d.mkdir(parents=True,exist_ok=True);assert not (d/'prepare.json').exists();reference,checker,proof=bind(case)
 # Original verifier AST and its exact numeric interpreter are preserved. No oracle or measured math runs here.
 assert proof['frozen_oracle_and_verification_FunctionDef_AST_unmodified'] and proof['original_normalized_bound']==1e-3
 case.__dict__.pop('inputs',None)
 with (d/'case.pkl').open('wb') as f:pickle.dump(case,f,protocol=5)
 os.link(old/'payload.pt',d/'payload.pt');assert sha(d/'payload.pt')==m['payload_sha256']
 sources={str(f):sha(f) for f in T.rglob('*') if f.is_file() and f.suffix in {'.py','.json','.yaml','.yml','.toml'} and '__pycache__' not in f.parts}
 for file in [Path(__file__),P/'selective_scan_denominator_frozen_policy.py',Path(proof['frozen_source']),Path(proof['frozen_numerics_source']),P/'TASKS.json',old/'prepare.json']:sources[str(file)]=sha(file)
 meta={'ordinal':i,'key':m['key'],'source_commit':'08f66b153c44b47330e7fe4311470152f287a112','torch':torch.__version__,'params':norm(case.params),'reference_policy':m['reference_policy'] if 'reference_policy' in m else 'original_npu','frozen_oracle_and_checker':proof,'original_prepare_path':str(old/'prepare.json'),'original_prepare_sha256':sha(old/'prepare.json'),'original_payload_byte_identical':True,'new_case_framework_only_preserves_cached_frozen_inputs_and_checker':True,'no_NPU_imported':True,'payload_sha256':sha(d/'payload.pt'),'descriptor_sha256':sha(d/'case.pkl'),'sources':sources};(d/'prepare.json').write_text(json.dumps(meta,indent=2));accepted.append({'ordinal':i,'prepare_sha256':sha(d/'prepare.json'),'payload_sha256':meta['payload_sha256'],'original_prepare_sha256':meta['original_prepare_sha256']});print('SCAN_DENOMINATOR_EXACT_ORIGINAL_PAYLOAD_POLICY_PREPARED',i,flush=True)
assert not any(n.split('.')[0] in ('torch_npu','acl') for n in sys.modules);(P/'SELECTIVE_SCAN_DENOMINATOR_CPU_PREPARATION.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_NPU_imported':True,'no_measured_or_oracle_math_repeated':True,'rows':accepted,'zero_native_qualification':True},indent=2))
