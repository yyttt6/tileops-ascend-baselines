"""CPU-only input/descriptor preparation. Oracle backend is NOT silently changed."""
import sys,os,pathlib,importlib.abc,json,time,pickle,hashlib,gc,shutil
P=pathlib.Path(__file__).resolve().parent;T=P/'workspace/TileOPs';tag=sys.argv[1];ids=list(map(int,sys.argv[2:]));assert len(ids)<=8
sys.path[:0]=[str(P),str(T/'src'),str(T)];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages');os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU preparation forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
from benchmarks.api import cases
from workloads.device import set_run_device
from workloads.numerics import describe
from reference_policy import policy
rows=json.loads((P/'TASKS.json').read_text());set_run_device('cpu');torch.set_num_threads(8);cache={}
# Registry packages declare CPU-only import; validate registration metadata before
# occupying the device. An actual device-only import stays for the guard.
from benchmarks.ascend_all_pairs.registry import LIBRARY_MODULES
lib=rows[ids[0]]['library']
if lib in LIBRARY_MODULES:
 try:__import__('importlib').import_module('benchmarks.ascend_all_pairs.'+LIBRARY_MODULES[lib])
 except ModuleNotFoundError as e:
  if e.name not in ('torch_npu','acl'):raise
 except FileNotFoundError as e:raise RuntimeError('CPU_REGISTRY_DEPENDENCY_MISSING: '+str(e)) from e

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
source={str(f):sha(f) for folder in [T/'benchmarks',T/'workloads',T/'src/tileops/manifest'] for f in folder.rglob('*.py')}
source[str(P/'reference_policy.py')]=sha(P/'reference_policy.py');source[str(P/'TASKS.json')]=sha(P/'TASKS.json')
for i in ids:
 r=rows[i];D=P/'prepared-rollout'/str(i)
 if (D/'prepare.json').exists():
  cached=json.loads((D/'prepare.json').read_text())
  if cached['sources']!=source:raise RuntimeError('CPU_PREPARE_CACHE_SOURCE_MISMATCH: preserve old metadata and explicitly review changed sources before device submission')
  assert sha(D/'payload.pt')==cached['payload_sha256'] and sha(D/'case.pkl')==cached['descriptor_sha256']
  continue
 t=time.perf_counter();torch.manual_seed(0)
 storage=pathlib.Path('/workspace/user_data/baseline-over3-20261110/ordered-inputs')/str(i);storage.mkdir(parents=True,exist_ok=True)
 if min(shutil.disk_usage('/').free,shutil.disk_usage(storage).free)<20_000_000_000:raise RuntimeError('disk20e9 floor; no predictive memory gate')
 if r['op'] not in cache:cache[r['op']]=list(cases(r['op']))
 template=next((c for c in cache[r['op']] if c.id==r['case']),None)
 if template is None:
  available={c.id for c in cache[r['op']]}
  missing=[j for j,row in enumerate(rows) if row['library']==r['library'] and row['op']==r['op'] and row['case'] not in available]
  print('FROZEN_CASE_UNAVAILABLE_JSON='+json.dumps(missing),flush=True)
  raise LookupError('Requested frozen exact case absent; do not hold other available cases of this operator')
 case=__import__('copy').copy(template);D.mkdir(parents=True,exist_ok=True)
 with (D/'case.pkl').open('wb') as f:pickle.dump(case,f,protocol=5)
 inputs=case.inputs
 target=storage/'payload.pt'
 if target.exists():raise FileExistsError('Preserve prior uncommitted payload; inspect before retry: '+str(target))
 torch.save({'inputs':inputs},target)
 (D/'payload.pt').symlink_to(target)
 metadata={'key':[r['library'],r['op'],r['case']],'params':case.params,'verification':describe(case.verification),'input_seed':0,'reference_policy':policy(r['library'],r['op'],r['case']),'expected_not_computed_on_wrong_device':True,'payload_storage':str(target),'direct_workspace_output_no_active_copy_moved':True,'payload_sha256':sha(D/'payload.pt'),'descriptor_sha256':sha(D/'case.pkl'),'torch':torch.__version__,'sources':source,'seconds':time.perf_counter()-t}
 (D/'prepare.json').write_text(json.dumps(metadata,indent=2));print('INPUT_READY',i,r['case'],round(metadata['seconds'],3),flush=True);del case,inputs;gc.collect()
