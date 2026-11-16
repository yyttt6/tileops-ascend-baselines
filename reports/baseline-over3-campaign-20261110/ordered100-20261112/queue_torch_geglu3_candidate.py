"""Exact three original published Torch programs, fresh source/input namespace and full checker."""
from pathlib import Path
import os,sys,json,datetime,fcntl,subprocess,shlex,time,hashlib
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_torch_geglu3_candidate import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=P/'TORCH_GEGLU3_CANDIDATE_EXPANSION.json';lock=(P/'torch-geglu3-candidate-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'source_deployment','scope':[1361,1362,1363],'paired_cases':0,'queued_at':now()}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
put()
try:
 assert not stop();prepared=json.loads((P/'TORCH_GELU3_ORIGINAL_CPU_PREPARATION.json').read_text());assert prepared['no_NPU_imported'] and len(prepared['prepared'])==3
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);b=P/'rollout_bootstrap.py';text=b.read_text()
  originals=json.loads((P/'TORCH_GELU3_ORIGINAL_CURRENT.json').read_text());assert originals['accepted_original_Torch_programs']==3
  before="'torch-gelu3-original-f037':'worker_torch_gelu3_original_f037_v2.py'"
  after="'torch-gelu3-original-f037':'worker_torch_geglu3_candidate.py'"
  assert text.count(before)==1,'Do not change unknown routing'
  (P/'server2-alignment-20261113/rollout_bootstrap.before-geglu3-candidate.py').write_text(text)
  text=text.replace(before,after);b.write_text(text)
  import ast
  mapping=next(n.value for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
  assert ast.literal_eval(mapping.func.value)['torch-gelu3-original-f037']=='worker_torch_geglu3_candidate.py','Actual route must use isolated runtime worker'
  files={}
  for name in ['rollout_bootstrap.py','worker_torch_geglu3_candidate.py','torch_gelu3_original_fidelity.py']:
   p=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=p.read_bytes(),check=True);h=hashlib.sha256(p.read_bytes()).hexdigest();assert c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==h;files[name]=h
  c.put(P/'TORCH_GEGLU3_CANDIDATE_DEPLOYMENT.json',{'checked_at':now(),'files':files,'complete_native_GeGlu_candidate_original_committed_checker':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'Complete_native_Torch_GeGlu3_tanh_same_original_contract','snapshot_read_only':True})
 for i in s['scope']:
  assert not stop();accepted=set(json.loads((P/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100'])
  assert i in accepted,'Complete original must be accepted before candidate'
  if (P/f'rollout-torch-geglu3-candidate-{i}-v1.CHECKPOINT.json').exists():continue
  assert not (P/f'TORCH_GEGLU3_CANDIDATE_{i}_HOLD.json').exists(),'No samecondition numeric/dependency retry'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','torch-gelu3-original-f037',[i],f'rollout-torch-geglu3-candidate-{i}-v1')
   if e is not None:break
   s.update(status='resource_deferred',current_ordinal=i);put();time.sleep(30)
  assert 'batch' in e,'Common guard/process fault, preserve and stop'
  r=e['batch']['results'][0]
  if r['status']!='raw_checked_pending_guard':c.put(P/f'TORCH_GEGLU3_CANDIDATE_{i}_HOLD.json',{'ordinal':i,'checkpoint':f'rollout-torch-gelu3-original-f037-{i}-v2.CHECKPOINT.json','result':r,'no_samecondition_retry':True});raise RuntimeError(('Original checker/dependency failure needs diagnosis',i,r))
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);summary=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);s.update(status='running',paired_cases=summary['paired_cases'],current_ordinal=i);put()
 s['status']='all3_complete_GeGlu_candidates_paired'
except BaseException as ex:s.update(status='stopped_requires_diagnosis',error=repr(ex));raise
finally:put();c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
