"""Exact three original published Torch programs, fresh source/input namespace and full checker."""
from pathlib import Path
import os,sys,json,datetime,fcntl,subprocess,shlex,time,hashlib
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_selective_scan_bare_original import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=P/'SELECTIVE_SCAN_BARE_ORIGINAL_EXPANSION.json';lock=(P/'selective-scan-bare-original-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'source_deployment','scope':[1499],'accepted_original_Torch_programs':0,'queued_at':now()}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
put()
try:
 assert not stop();prepared=json.loads((P/'SELECTIVE_SCAN_BARE_ORIGINAL_CPU_PREPARATION.json').read_text());assert prepared['no_NPU_imported'] and prepared['ordinal']==1499 and prepared['frozen_reference_checker_preserved']
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);b=P/'rollout_bootstrap.py';text=b.read_text()
  import ast
  mapping=next(n.value for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
  assert ast.literal_eval(mapping.func.value)['torch-selective-scan-bare']=='worker_selective_scan_bare_original.py','Actual route must use isolated runtime worker'
  files={}
  for name in ['rollout_bootstrap.py','worker_selective_scan_bare_original.py','selective_scan_bare_fidelity.py','selective_scan_bare_device.py','selective_scan_bare_frozen_policy.py','SELECTIVE_SCAN_BARE_PUBLISHED_ROW.json','prepare_selective_scan_bare_original_cpu.py']:
   p=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=p.read_bytes(),check=True);h=hashlib.sha256(p.read_bytes()).hexdigest();assert c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==h;files[name]=h
  c.put(P/'SELECTIVE_SCAN_BARE_ORIGINAL_DEPLOYMENT.json',{'checked_at':now(),'files':files,'original_storage_Torch_entry_and_original_committed_checker':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'Restored_bare_scan_original08f_device_program_frozen_FP64_oracle_and_checker_full15','snapshot_read_only':True})
 for i in s['scope']:
  assert not stop();accepted=set(json.loads((P/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100'])
  if i in accepted:continue
  assert not (P/f'SELECTIVE_SCAN_BARE_ORIGINAL_{i}_HOLD.json').exists(),'No samecondition numeric/dependency retry'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','torch-selective-scan-bare',[i],f'rollout-torch-selective-scan-bare-{i}-v3')
   if e is not None:break
   s.update(status='resource_deferred',current_ordinal=i);put();time.sleep(30)
  assert 'batch' in e,'Common guard/process fault, preserve and stop'
  r=e['batch']['results'][0]
  if r['status']!='raw_checked_pending_guard':c.put(P/f'SELECTIVE_SCAN_BARE_ORIGINAL_{i}_HOLD.json',{'ordinal':i,'checkpoint':f'rollout-torch-selective-scan-bare-{i}-v2.CHECKPOINT.json','result':r,'no_samecondition_retry':True});raise RuntimeError(('Original checker/dependency failure needs diagnosis',i,r))
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);summary=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);s.update(status='running',accepted_original_Torch_programs=summary['accepted_original_Torch_programs'],current_ordinal=i);put()
 s['status']='restored_bare_scan_program_formal_verified'
except BaseException as ex:s.update(status='stopped_requires_diagnosis',error=repr(ex));raise
finally:put();c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
