"""Sequential independent H1 continuation; same complete Torch math/inputdevice."""
from pathlib import Path
import json,datetime,fcntl,os,sys,subprocess,shlex,hashlib,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_torch_fusedqk_remaining import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'torch-fusedqk-remaining.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'queued_at':now(),'status':'CPU_preparation','scope':list(range(1356,1361)),'accepted_original_APIs':0,'original_CPU_oracle_unchanged':True,'not_paired_speedup_or_public_update':True};c.put(P/'TORCH_FUSEDQK_REMAINING_EXPANSION.json',state)
try:
 assert not stop();prior=json.loads((P/'OPS_NN_UNARY_EXPANSION.json').read_text());assert prior['status']=='all10_exact_original_APIs_fullchecker15raw_guard_valid' and prior['accepted_original_APIs']==10
 proof=json.loads((P/'TORCH_FUSEDQK_REMAINING_CPU_PROOF.json').read_text());assert proof['CPU_cases']==24 and proof['helper_sha256']==hashlib.sha256((P/'torch_fusedqk_remaining_reference.py').read_bytes()).hexdigest()
 ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text());assert all(ledger[str(i)]['result']['status']=='provider_dependency_deferred' and ledger[str(i)]['result']['cause']['no_device_attempt'] for i in state['scope'])
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'CPU_frozen_input_prepare_then_complete_Torch_FusedQK_originalisolated15','durable_log':'torch-fusedqk-remaining-queued.log','snapshot_read_only':True})
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);f=P/'rollout_bootstrap.py';s=f.read_text()
  if "'torch-fusedqk-remaining':" not in s:f.with_name('rollout_bootstrap.before-fusedqk-remaining.py').write_text(s);f.write_text(s.replace('script={',"script={'torch-fusedqk-remaining':'worker_torch_fusedqk_remaining.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','worker_torch_fusedqk_remaining.py','torch_fusedqk_remaining_reference.py','torch_fusedqk_remaining_cached_fidelity.py','verify_torch_fusedqk_remaining_cached.py']:
   f=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);actual=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=actual
  c.put(P/'TORCH_FUSEDQK_REMAINING_DEPLOYMENT.json',{'checked_at':now(),'files':records,'original_CPU_oracle_unchanged':True,'placement_only_full_reverse_AST':True,'CPU_boundary_cases':24})
 for i in state['scope']:
  assert not stop()
  with (P/f'torch-fusedqk-remaining-{i}-prepare.CPU.log').open('w') as out:rc=c.remote('TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+c.R+'/prepare_rollout.py torch-fusedqk-remaining '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'CPU case/source fault; no sharedscope device retries'
  with (P/f'torch-fusedqk-remaining-{i}-verify.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_torch_fusedqk_remaining_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'Frozen source/payload fault'
  label='rollout-torch-fusedqk-remaining-'+str(i)+'-v1'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','torch-fusedqk-remaining',[i],label)
   if e is not None:break
   state.update(status='resource_deferred',checked_at=now());c.put(P/'TORCH_FUSEDQK_REMAINING_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in e,'Guard/process fault archived; no blindretry'
  assert all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']),'Full checker/binding failure retained; diagnose first'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=[i],accepted_original_APIs=report['accepted_original_APIs'],checked_at=now());c.put(P/'TORCH_FUSEDQK_REMAINING_EXPANSION.json',state)
 state['status']='all5_exact_original_Torch_FusedQK_fullchecker15raw_guard_valid'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=now();c.put(P/'TORCH_FUSEDQK_REMAINING_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
