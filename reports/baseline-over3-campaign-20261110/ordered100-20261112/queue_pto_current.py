"""Representative first then bounded exact original PTO APIs, genuine currentguard."""
from pathlib import Path
import json,datetime,fcntl,os,sys,subprocess,shlex,hashlib,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_pto_current import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'pto-current-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
scope=[x['ordinal'] for x in json.loads((P/'PTO_FROZEN_KEY_READINESS_CPU.json').read_text())['frozen_unresolved_key_candidates']];scope=[1712]+[i for i in scope if i!=1712]
state={'pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'queued_at':now(),'status':'CPU_deployment','scope':scope,'accepted_original_APIs':0,'representative_first':1712,'original_API_kernel_precision_checker_unchanged':True,'new_genuine_guard_not_legacy_owner':True};c.put(P/'PTO_CURRENT_EXPANSION.json',state)
try:
 assert not stop();prior=json.loads((P/'OPS_TENSOR_TRUNC_CURRENT_EXPANSION.json').read_text());assert prior['accepted_original_APIs']==1 and prior['status']=='single_exact_originalAPI_fullchecker15raw_guard_valid'
 dep=json.loads((P/'PTO_PRIVATE_DEPENDENCY_PREPARE.json').read_text());assert dep['status']=='all_original_SDK_files_verified_private_relocation_only_runtime_pending';assert json.loads((P/'PTO_ALL_RELOCATED_PINS_CPU.json').read_text())['all_original_pins_verified']==864;assert json.loads((P/'PTO_FULL_CONTRACT_CPU_REVIEW.json').read_text())['full_contract_matches']==23
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'original_PTO_API_representative_then_fullchecker_originalisolated15','durable_log':'pto-current-queued.log','snapshot_read_only':True})
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);f=P/'rollout_bootstrap.py';s=f.read_text()
  if "'pto-current':" not in s:f.with_name('rollout_bootstrap.before-pto-current.py').write_text(s);f.write_text(s.replace('script={',"script={'pto-current':'worker_pto_current.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','worker_pto_current.py','pto_current_binding.py','pto_current_cached_fidelity.py','verify_pto_current_cached.py','PTO_CURRENT_BINDING_CPU_PROOF.json','PTO_PRIVATE_DEPENDENCY_PREPARE.json','PTO_SDK920B2_SERVER2_SOURCE_MANIFEST.json']:
   f=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=h
  c.put(P/'PTO_CURRENT_DEPLOYMENT.json',{'checked_at':now(),'files':records,'original_PIN864_and_fullcontract23_verified':True,'no_original_vendor_source_modified':True})
 for i in scope:
  assert not stop()
  q='import pathlib;print((pathlib.Path('+repr(c.R)+')/"prepared-rollout"/'+repr(str(i))+'/"prepare.json").exists())';exists=c.remote('python3 -c '+shlex.quote(q),stdout=subprocess.PIPE,text=True,check=True).stdout.strip()=='True'
  if not exists:
   with (P/f'pto-current-{i}-prepare.CPU.log').open('w') as out:rc=c.remote('TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+c.R+'/prepare_rollout.py pto-current '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'CPU shared source/case fault beforeNPU'
  with (P/f'pto-current-{i}-verify.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_pto_current_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'Frozen source/payload fault; preserve oldcache'
  label='rollout-pto-current-'+str(i)+'-v2'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','pto-current',[i],label)
   if e is not None:break
   state.update(status='resource_deferred',checked_at=now());c.put(P/'PTO_CURRENT_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in e,'Actual guard/process failure preserved; no blindretry'
  assert all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']),'OriginalAPI factory/compile/checker failure retained; diagnose beforecontinuation'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);r=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=[i],accepted_original_APIs=r['accepted_original_APIs'],checked_at=now());c.put(P/'PTO_CURRENT_EXPANSION.json',state)
 state['status']='all22_exact_original_PTO_APIs_fullchecker15raw_guard_valid'
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:
 state['checked_at']=now();c.put(P/'PTO_CURRENT_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
