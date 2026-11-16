"""Original Torch cases alreadyCPUprepared but neverexecuted after latercatalog failure."""
import pathlib,json,sys,os,datetime,fcntl,hashlib,subprocess,shlex,time
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_torch_fusedqk_device import summarize
lock=(P/'torch-fusedqk-device.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
state={'queued_at':stamp(),'pid':os.getpid(),'host':c.H,'status':'CPU_preflight','indices':[1354,1355],'accepted_original_APIs':0,'complete_original_Torch_math_device_placement':True,'original_CPU_oracle_unchanged':True,'no_alternate_or_ratio_claim':True};c.put(P/'TORCH_DEVICE_FUSEDQK_EXPANSION.json',state)
try:
 assert not stop();old=json.loads((P/'TORCH_READY_ORIGINAL_EXPANSION.json').read_text());assert old['status']=='partial_requires_diagnosis_no_samecondition_retry' and old['accepted_original_APIs']==0
 prior=json.loads((P/'VLLM_NORM_FLA_DEFINED_EXPANSION.json').read_text());assert prior['status']=='all_exact_scope_attempted_preserve_failures'
 try:assert b'queue_vllm_norm_fla_defined.py' not in pathlib.Path('/proc/1553447/cmdline').read_bytes()
 except FileNotFoundError:pass
 ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text());assert all(ledger[str(i)]['result']['status']=='provider_dependency_deferred' and ledger[str(i)]['result']['cause']['no_device_attempt'] for i in state['indices'])
 proof=json.loads((P/'TORCH_FUSEDQK_DEVICE_CPU_PROOF.json').read_text());assert proof['CPU_cases']==12 and proof['proof']['full_reverse_AST_identical'];assert proof['helper_sha256']==hashlib.sha256((P/'torch_fusedqk_device_reference.py').read_bytes()).hexdigest()
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);assert not stop()
  f=P/'rollout_bootstrap.py';s=f.read_text();f.with_name('rollout_bootstrap.before-torch-ready.py').write_text(s)
  if "'torch-fusedqk-device':" not in s:f.write_text(s.replace('script={',"script={'torch-fusedqk-device':'worker_torch_fusedqk_device.py',",1))
  f=P/'continue_ordered.py';s=f.read_text();needle="'rollout-torch-ready-fusedqk-'";assert needle in s
  if "'rollout-torch-device-fusedqk-'" not in s:f.with_name('continue_ordered.before-torch-ready.py').write_text(s);f.write_text(s.replace(needle,needle+",'rollout-torch-device-fusedqk-'",1))
  f=P/'effective_coverage.py';s=f.read_text();needle="'RESOLVED_TORCH_READY_ORIGINAL.json'";assert needle in s
  if "'RESOLVED_TORCH_DEVICE_FUSEDQK.json'" not in s:f.with_name('effective_coverage.before-torch-ready.py').write_text(s);f.write_text(s.replace(needle,needle+",'RESOLVED_TORCH_DEVICE_FUSEDQK.json'",1))
  records=[]
  for name in ['rollout_bootstrap.py','torch_ready_cached_fidelity.py','verify_torch_ready_cached.py','worker_torch_fusedqk_device.py','torch_fusedqk_device_reference.py']:
   f=P/name;dst=c.R+'/'+name;code='import pathlib,sys;pathlib.Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);actual=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==hashlib.sha256(f.read_bytes()).hexdigest();records.append({'path':name,'sha256':actual})
  c.put(P/'TORCH_DEVICE_FUSEDQK_DEPLOYMENT.json',{'checked_at':stamp(),'files':records,'full_math_AST_unchanged_except_CPU_oracle_placement':True,'CPU_proof':'TORCH_FUSEDQK_DEVICE_CPU_PROOF.json','timing_checker_math_unchanged':True,'old_catalog_failure_after_two_INPUT_READY_preserved':True})
 for i in state['indices']:
  with (P/f'torch-ready-{i}-CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_torch_ready_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'Source/input qualification failed, no device submission'
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','updated_at':stamp(),'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'phase':'original_Torch_frozen_ready_two_cases_isolated15','durable_log':'torch-fusedqk-device-queued.log','snapshot_read_only':True});state.update(status='running',started_at=stamp());c.put(P/'TORCH_DEVICE_FUSEDQK_EXPANSION.json',state)
 label='rollout-torch-device-fusedqk-1354-1355-v1'
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);event=c.run('r-template','torch-fusedqk-device',state['indices'],label)
  if event is not None:break
  state.update(status='resource_deferred',checked_at=stamp());c.put(P/'TORCH_DEVICE_FUSEDQK_EXPANSION.json',state);time.sleep(30)
 assert 'batch' in event,'Guard/process failure retained; no blindretry'
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='all_attempted_preserve_failures' if len(event['batch']['results'])==2 else 'partial_requires_diagnosis_no_samecondition_retry',accepted_original_APIs=report['accepted_original_APIs'],results=[{'ordinal':r['ordinal'],'status':r['status'],'phase':r.get('phase'),'error':r.get('error')} for r in event['batch']['results']])
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:
 state['checked_at']=stamp();c.put(P/'TORCH_DEVICE_FUSEDQK_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','checked_at':stamp(),'pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'outcome':state['status'],'next_action':'Auditactualscope; handle commonfault once, continue qualified independent nextscope','snapshot_read_only':True})
