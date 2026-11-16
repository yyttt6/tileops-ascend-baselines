"""Exact three original published Torch programs, fresh source/input namespace and full checker."""
from pathlib import Path
import os,sys,json,datetime,fcntl,subprocess,shlex,time,hashlib
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_torch_gelu3_original import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=P/'TORCH_GELU3_ORIGINAL_EXPANSION.json';lock=(P/'torch-gelu3-original-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'source_deployment','scope':[1361,1362,1363],'accepted_original_Torch_programs':0,'queued_at':now()}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
put()
try:
 assert not stop();prepared=json.loads((P/'TORCH_GELU3_ORIGINAL_CPU_PREPARATION.json').read_text());assert prepared['no_NPU_imported'] and len(prepared['prepared'])==3
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);b=P/'rollout_bootstrap.py';text=b.read_text()
  if "'torch-gelu3-original-f037':" not in text:(P/'rollout_bootstrap.before-torch-gelu3-original.py').write_text(text);b.write_text(text.replace('script={',"script={'torch-gelu3-original-f037':'worker_torch_gelu3_original_f037.py',",1))
  files={}
  for name in ['rollout_bootstrap.py','worker_torch_gelu3_original_f037.py','torch_gelu3_original_fidelity.py']:
   p=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=p.read_bytes(),check=True);h=hashlib.sha256(p.read_bytes()).hexdigest();assert c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==h;files[name]=h
  c.put(P/'TORCH_GELU3_ORIGINAL_DEPLOYMENT.json',{'checked_at':now(),'files':files,'original_storage_Torch_entry_and_original_committed_checker':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'Exact_f037_original_Torch_GELU3_storage_dtype_tanh_fullchecker_isolated15','snapshot_read_only':True})
 for i in s['scope']:
  assert not stop();accepted=set(json.loads((P/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100'])
  if i in accepted:continue
  assert not (P/f'TORCH_GELU3_ORIGINAL_{i}_HOLD.json').exists(),'No samecondition numeric/dependency retry'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','torch-gelu3-original-f037',[i],f'rollout-torch-gelu3-original-f037-{i}-v1')
   if e is not None:break
   s.update(status='resource_deferred',current_ordinal=i);put();time.sleep(30)
  assert 'batch' in e,'Common guard/process fault, preserve and stop'
  r=e['batch']['results'][0]
  if r['status']!='raw_checked_pending_guard':c.put(P/f'TORCH_GELU3_ORIGINAL_{i}_HOLD.json',{'ordinal':i,'checkpoint':f'rollout-torch-gelu3-original-f037-{i}-v1.CHECKPOINT.json','result':r,'no_samecondition_retry':True});raise RuntimeError(('Original checker/dependency failure needs diagnosis',i,r))
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);summary=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);s.update(status='running',accepted_original_Torch_programs=summary['accepted_original_Torch_programs'],current_ordinal=i);put()
 s['status']='all3_original_Torch_GELU_tanh_programs_formal_verified'
except BaseException as ex:s.update(status='stopped_requires_diagnosis',error=repr(ex));raise
finally:put();c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
