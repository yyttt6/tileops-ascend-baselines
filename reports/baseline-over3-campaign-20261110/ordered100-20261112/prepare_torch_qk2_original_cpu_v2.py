"""Exact original f037 torch-program contract/input preparation; CPU-only, no aliases."""
from pathlib import Path
import os,sys,importlib.abc,json,hashlib,datetime,pickle,gc
P=Path(__file__).resolve().parent;T=P/'workspace/TileOps-08f-original';sys.path[:0]=[str(T/'src'),str(T),str(P)];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU source preparation forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
from benchmarks.api import cases
from workloads.device import set_run_device
from workloads.numerics import compare_outputs,describe
set_run_device('cpu');torch.set_num_threads(8);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=json.loads((P/'TASKS.json').read_text());pub=json.loads((P/'TORCH_QK2_PUBLISHED_ROWS.json').read_text());configs={r['case']:r for r in pub};source={str(f):sha(f) for f in T.rglob('*') if f.is_file() and f.suffix in {'.py','.json','.yaml','.yml','.toml'} and '__pycache__' not in f.parts};source[str(P/'TASKS.json')]=sha(P/'TASKS.json');source[str(Path(__file__))]=sha(Path(__file__));allcases=cases('FusedQKNormRoPEFwdOp');prepared=[]
def norm(x):
 if isinstance(x,torch.dtype):return str(x).removeprefix('torch.')
 if isinstance(x,(tuple,list)):return [norm(v) for v in x]
 if isinstance(x,dict):return {k:norm(v) for k,v in x.items()}
 return x
for i in [1356,1358]:
 row=rows[i];assert row['library']=='torch-eager' and row['op']=='FusedQKNormRoPEFwdOp';case=next(x for x in allcases if x.id==row['case']);assert case._call.signature.name==row['op'];original=configs[row['case']];assert norm(case.params)==original['workload']['params'],(norm(case.params),original['workload']['params'])
 d=P/'prepared-torch-qk2-08f-v2'/str(i);d.mkdir(parents=True,exist_ok=True);assert not (d/'prepare.json').exists(),'Never overwrite accepted cached input';torch.manual_seed(0)
 with (d/'case.pkl').open('wb') as f:pickle.dump(case,f,protocol=5)
 xs=case.inputs;before=[x.contiguous().view(torch.uint8).clone() for x in xs];expected=case.reference(*xs);torch_storage=case.reference(*xs);check=compare_outputs(torch_storage,expected,case.verification);assert check.checked_outputs==check.total_outputs and not check.unchecked_reason;assert all(torch.equal(x.contiguous().view(torch.uint8),b) for x,b in zip(xs,before));torch.save({'inputs':xs,'expected':expected},d/'payload.pt');meta={'ordinal':i,'key':[row['library'],row['op'],row['case']],'source_commit':'08f66b153c44b47330e7fe4311470152f287a112','params':norm(case.params),'seed':0,'torch':torch.__version__,'measured_entry':'case.reference (original08f complete device-resident Torch program)','oracle_entry':'original case.reference, original committed verification','verification':describe(case.verification),'CPU_original_Torch_vs_original_reference_passed':True,'all_inputs_bit_preserved':True,'no_NPU_imported':not any(x.split('.')[0] in ('torch_npu','acl') for x in sys.modules),'payload_sha256':sha(d/'payload.pt'),'descriptor_sha256':sha(d/'case.pkl'),'sources':source};assert meta['no_NPU_imported'];(d/'prepare.json').write_text(json.dumps(meta,indent=2)+'\n');prepared.append({'ordinal':i,'prepare_sha256':sha(d/'prepare.json'),'payload_sha256':meta['payload_sha256'],'input_bytes':sum(x.numel()*x.element_size() for x in xs)});print('ORIGINAL_QK_CPU_PREPARED',i,meta['measured_entry'],flush=True);case._release();del xs,expected,torch_storage,before;gc.collect()
assert all(sha(p)==h for p,h in source.items());(P/'TORCH_QK2_ORIGINAL_CPU_PREPARATION_V2.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_NPU_imported':True,'source_commit':'08f66b153c44b47330e7fe4311470152f287a112','prepared':prepared,'zero_formal_coverage':True},indent=2)+'\n')
