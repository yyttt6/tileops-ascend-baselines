"""Queue missing Norm coverage via a different complete vendor API, no old retries."""
import pathlib,sys,json,time,fcntl,subprocess,shlex,hashlib,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_torch_extended_device import summarize
lock=(P/'vllm-torch-extended-device.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
def predecessor_running():
 try:return b'queue_torch_fusedqk_device.py' in pathlib.Path('/proc/1559728/cmdline').read_bytes()
 except FileNotFoundError:return False
state={'queued_at':stamp(),'status':'CPU_wait_for_predecessor','indices':[1296,1348,1349,1350,1351,1352,1353],'batches':[[1296],[1348],[1349,1350,1351,1352,1353]],'pid':os.getpid(),'host':c.H,'representative':1348,'accepted_original_APIs':0,'full_Torch_math_only_device_placement_changed':True,'CPU_oracle_unchanged':True,'old_CPU_placement_failures_retained':True,'effective_coverage_addition_only_after_audit':True};c.put(P/'TORCH_EXTENDED_DEVICE_EXPANSION.json',state)
try:
 while predecessor_running():assert not stopped();time.sleep(5)
 assert not stopped()
 prior=json.loads((P/'TORCH_DEVICE_FUSEDQK_EXPANSION.json').read_text());assert prior['status']=='all_attempted_preserve_failures' and prior['accepted_original_APIs']==2,'Review shared/runtime fault before successor'
 proof=json.loads((P/'TORCH_EXTENDED_DEVICE_CPU_PROOF.json').read_text());assert proof['CPU_cases']==24 and all(v['full_reverse_AST_identical'] for v in proof['proofs'].values())
 assert proof['helper_sha256']==hashlib.sha256((P/'torch_extended_device_reference.py').read_bytes()).hexdigest()
 ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text())
 for i in state['indices']:assert 'Missing named task coverage records' in ledger[str(i)]['result'].get('error','')
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);assert not stopped()
  f=P/'rollout_bootstrap.py';old=f.read_text();s=old
  if "'torch-extended-device':" not in s:s=s.replace('script={',"script={'torch-extended-device':'worker_torch_extended_device.py',",1)
  f.with_name('rollout_bootstrap.before-torch-extended-device.py').write_text(old);f.write_text(s)
  f=P/'continue_ordered.py';old=f.read_text();s=old;needle="'rollout-torch-device-fusedqk-'";assert needle in s
  if "'rollout-vllm-torch-extended-device-'" not in s:s=s.replace(needle,needle+",'rollout-vllm-torch-extended-device-'",1)
  f.with_name('continue_ordered.before-torch-extended-device.py').write_text(old);f.write_text(s)
  f=P/'effective_coverage.py';old=f.read_text();needle="'RESOLVED_TORCH_DEVICE_FUSEDQK.json'";assert needle in old
  if "'RESOLVED_TORCH_EXTENDED_DEVICE.json'" not in old:
   f.with_name('effective_coverage.before-torch-extended-device.py').write_text(old);f.write_text(old.replace(needle,needle+",'RESOLVED_TORCH_EXTENDED_DEVICE.json'",1))
  records=[]
  for rel in ['rollout_bootstrap.py','torch_extended_cached_fidelity.py','verify_torch_extended_cached.py','worker_torch_extended_device.py','torch_extended_device_reference.py']:
   f=P/rel;dst=c.R+'/'+rel;code='import pathlib,sys;p=pathlib.Path('+repr(dst)+');p.write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);actual=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==hashlib.sha256(f.read_bytes()).hexdigest();records.append({'path':rel,'sha256':actual})
  c.put(P/'TORCH_EXTENDED_DEVICE_DEPLOYMENT.json',{'checked_at':stamp(),'files':records,'CPU_proof':'TORCH_EXTENDED_DEVICE_CPU_PROOF.json','source_provenance':'TORCH_EXTENDED_DEVICE_CPU_PROOF.json','protocol':'ascend950-isolated-ffts-v3','original_global_guard_unchanged':True,'no_old_control_repeated':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','updated_at':stamp(),'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'phase':'complete_original_Torch_math_deviceplacement_CPUoracle_separate_originalisolated15','durable_log':'torch-extended-device-queued.log','snapshot_read_only':True});state.update(status='running',started_at=stamp());c.put(P/'TORCH_EXTENDED_DEVICE_EXPANSION.json',state)
 pending=list(state['batches']);serial=0
 while pending:
  ids=pending.pop(0);serial+=1
  assert not stopped()
  for i in ids:
   with (P/f'torch-extended-device-{i}-cached-source.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_torch_extended_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'Cached source/payload qualification failed; no percase spin'
  label='rollout-vllm-torch-extended-device-'+'-'.join(map(str,ids))+'-v1'
  while True:
   assert not stopped()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);event=c.run('r-template','torch-extended-device',ids,label)
   if event is not None:break
   state.update(status='resource_deferred',pending_label=label,checked_at=stamp());c.put(P/'TORCH_EXTENDED_DEVICE_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in event,'Invalid guard/process event preserved; requires diagnosis'
  for r in event['batch']['results']:
   if r.get('phase')=='factory' and not r.get('error','').startswith('UnsupportedCase('):raise RuntimeError('Unexpected factory/binding fault; stop shared group')
   if 'Missing named task coverage records' in r.get('error',''):raise RuntimeError('Common deviceplacement/profilefault remains; stop before percase spin')
   if r.get('phase')=='load':raise RuntimeError('Shared load/source fault; stop affected group')
   if r['status']!='raw_checked_pending_guard':state.setdefault('failures',[]).append({'ordinal':r['ordinal'],'checkpoint':label+'.CHECKPOINT.json'})
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=ids,accepted_original_APIs=report['accepted_original_APIs'],checked_at=stamp());c.put(P/'TORCH_EXTENDED_DEVICE_EXPANSION.json',state)
  attempted={r['ordinal'] for r in event['batch']['results']};assert attempted and attempted.issubset(set(ids))
  rest=[i for i in ids if i not in attempted]
  if rest:pending.insert(0,rest)
  if state['representative'] in ids:assert any(r['ordinal']==state['representative'] for r in report['rows']),'Representative failed; no expansion/retry'
 state['status']='all_exact_scope_attempted_preserve_failures'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=stamp();c.put(P/'TORCH_EXTENDED_DEVICE_EXPANSION.json',state)
 if not predecessor_running():c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','checked_at':stamp(),'pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'outcome':state['status'],'next_action':'Review exact alternative coverage; dispatch qualified successor. Do not invent old/new ratios.','snapshot_read_only':True})
