"""Original Torch cases alreadyCPUprepared but neverexecuted after latercatalog failure."""
import pathlib,json,sys,os,datetime,fcntl,hashlib,subprocess,shlex,time
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_tilelang_guard_recovery import summarize
lock=(P/'tilelang-guard-recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
state={'queued_at':stamp(),'pid':os.getpid(),'host':c.H,'status':'CPU_preflight','indices':[1587,1588,1589,1590],'accepted_original_APIs':0,'only_original_TileLang_API_measured':True,'no_alternate_or_ratio_claim':True};c.put(P/'TILELANG_GUARD_RECOVERY_EXPANSION.json',state)
try:
 assert not stop();prior=json.loads((P/'TORCH_EXTENDED_DEVICE_EXPANSION.json').read_text());assert prior['status']=='all_exact_scope_attempted_preserve_failures' and prior['accepted_original_APIs']==7
 try:assert b'queue_torch_extended_device.py' not in pathlib.Path('/proc/1565822/cmdline').read_bytes()
 except FileNotFoundError:pass
 old=json.loads((P/'rollout-1587-1590.FAILED.json').read_text());assert old['guard']['phase']=='finished' and old['guard']['workload_exit_code']==0 and old['guard']['performance_valid'] is False
 assert old['indices']==state['indices']
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);assert not stop()
  f=P/'rollout_bootstrap.py';s=f.read_text();f.with_name('rollout_bootstrap.before-tilelang-recovery.py').write_text(s)
  if "'tilelang-guard-recovery':" not in s:f.write_text(s.replace('script={',"script={'tilelang-guard-recovery':'worker_tilelang_guard_recovery.py',",1))
  f=P/'continue_ordered.py';s=f.read_text();needle="'rollout-vllm-torch-extended-device-'";assert needle in s
  if "'rollout-tilelang-guard-recovery-'" not in s:f.with_name('continue_ordered.before-tilelang-recovery.py').write_text(s);f.write_text(s.replace(needle,needle+",'rollout-tilelang-guard-recovery-'",1))
  f=P/'effective_coverage.py';s=f.read_text();needle="'RESOLVED_TORCH_EXTENDED_DEVICE.json'";assert needle in s
  if "'RESOLVED_TILELANG_GUARD_RECOVERY.json'" not in s:f.with_name('effective_coverage.before-tilelang-recovery.py').write_text(s);f.write_text(s.replace(needle,needle+",'RESOLVED_TILELANG_GUARD_RECOVERY.json'",1))
  records=[]
  for name in ['rollout_bootstrap.py','tilelang_recovery_cached_fidelity.py','verify_tilelang_recovery_cached.py','worker_tilelang_guard_recovery.py']:
   f=P/name;dst=c.R+'/'+name;code='import pathlib,sys;pathlib.Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);actual=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==hashlib.sha256(f.read_bytes()).hexdigest();records.append({'path':name,'sha256':actual})
  c.put(P/'TILELANG_GUARD_RECOVERY_DEPLOYMENT.json',{'checked_at':stamp(),'files':records,'original_native_worker_only_sourcefidelity_and_originalisolated_timer_selected':True,'original_isolated_timer_checker_and_math_unchanged':True,'old_invalid_guard_and_profiles_retained':True})
 for i in state['indices']:
  with (P/f'torch-ready-{i}-CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_tilelang_recovery_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'Source/input qualification failed, no device submission'
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','updated_at':stamp(),'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'phase':'original_TileLang_RMSNorm4_guard_invalid_recovery_isolated15','durable_log':'tilelang-guard-recovery-queued.log','snapshot_read_only':True});state.update(status='running',started_at=stamp());c.put(P/'TILELANG_GUARD_RECOVERY_EXPANSION.json',state)
 label='rollout-tilelang-guard-recovery-1587-1590-v1'
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);event=c.run('r-tilelang-original','tilelang-guard-recovery',state['indices'],label)
  if event is not None:break
  state.update(status='resource_deferred',checked_at=stamp());c.put(P/'TILELANG_GUARD_RECOVERY_EXPANSION.json',state);time.sleep(30)
 assert 'batch' in event,'Guard/process failure retained; no blindretry'
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='all_attempted_preserve_failures' if len(event['batch']['results'])==4 else 'partial_requires_diagnosis_no_samecondition_retry',accepted_original_APIs=report['accepted_original_APIs'],results=[{'ordinal':r['ordinal'],'status':r['status'],'phase':r.get('phase'),'error':r.get('error')} for r in event['batch']['results']])
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:
 state['checked_at']=stamp();c.put(P/'TILELANG_GUARD_RECOVERY_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','checked_at':stamp(),'pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'outcome':state['status'],'next_action':'Auditactualscope; handle commonfault once, continue qualified independent nextscope','snapshot_read_only':True})
