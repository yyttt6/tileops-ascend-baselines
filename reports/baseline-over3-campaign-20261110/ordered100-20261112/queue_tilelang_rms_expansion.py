"""Previouslydeferred original TileLangRMS18; no validcases or numericalfailures repeated."""
import pathlib,sys,json,os,subprocess,shlex,datetime,time,fcntl,hashlib
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_tilelang_rms_expansion import summarize
lock=(P/'tilelang-rms-expansion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
def previous_alive():
 try:return b'queue_tilelang_guard_recovery.py' in pathlib.Path('/proc/1568857/cmdline').read_bytes()
 except FileNotFoundError:return False
state={'pid':os.getpid(),'host':c.H,'queued_at':now(),'status':'CPU_wait_for_qualified_predecessor','indices':list(range(1591,1609)),'accepted_original_APIs':0,'no_predictive_memory_gate':True};c.put(P/'TILELANG_RMS_EXPANSION.json',state)
try:
 while previous_alive():assert not stop();time.sleep(5)
 assert not stop();prior=json.loads((P/'TILELANG_GUARD_RECOVERY_EXPANSION.json').read_text());assert prior['accepted_original_APIs']==4 and prior['status']=='all_attempted_preserve_failures','Review precursoractualfault beforeexpansion'
 ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text());assert all(ledger[str(i)]['result']['status']=='provider_dependency_deferred' and ledger[str(i)]['result']['cause']['evidence']=='rollout-1587-1590.FAILED.json' for i in state['indices'])
 assert (P/'worker_tilelang_rms_expansion.py').read_bytes().replace(b'tilelang_expansion_cached_fidelity',b'tilelang_recovery_cached_fidelity')==(P/'worker_tilelang_guard_recovery.py').read_bytes()
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);assert not stop()
  for name,needle,addition in [('rollout_bootstrap.py','script={',"script={'tilelang-rms-expansion':'worker_tilelang_rms_expansion.py',"),('continue_ordered.py',"'rollout-tilelang-guard-recovery-'","'rollout-tilelang-guard-recovery-','rollout-tilelang-rms-expansion-'"),('effective_coverage.py',"'RESOLVED_TILELANG_GUARD_RECOVERY.json'","'RESOLVED_TILELANG_GUARD_RECOVERY.json','RESOLVED_TILELANG_RMS_EXPANSION.json'")]:
   f=P/name;s=f.read_text();assert needle in s
   if ("'tilelang-rms-expansion':" if name=='rollout_bootstrap.py' else addition) not in s:
    b=f.with_name(f.stem+'.before-tilelang-rms-expansion.py');assert not b.exists();b.write_text(s);f.write_text(s.replace(needle,addition,1))
  records=[]
  for name in ['rollout_bootstrap.py','tilelang_expansion_cached_fidelity.py','verify_tilelang_expansion_cached.py','worker_tilelang_rms_expansion.py']:
   f=P/name;dst=c.R+'/'+name;code='import pathlib,sys;pathlib.Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records.append({'path':name,'sha256':h})
  c.put(P/'TILELANG_RMS_EXPANSION_DEPLOYMENT.json',{'checked_at':now(),'files':records,'original_complete_API_worker_math_and_isolated_timer_unchanged':True})
 c.put(P/'CURRENT_PROCESS.json',{'pid':os.getpid(),'status':'active','entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'original_TileLang_RMSNorm18_previouslydeferred','durable_log':'tilelang-rms-expansion-queued.log','snapshot_read_only':True});state.update(status='running',started_at=now());c.put(P/'TILELANG_RMS_EXPANSION.json',state)
 pending=[list(range(i,min(i+6,1609))) for i in range(1591,1609,6)]
 while pending:
  ids=pending.pop(0);assert not stop()
  code='import pathlib,json;print(json.dumps([i for i in '+repr(ids)+' if not (pathlib.Path('+repr(c.R)+')/\"prepared-rollout\"/str(i)/\"prepare.json\").exists()]))';missing=json.loads(c.remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if missing:
   log='tilelang-rms-expansion-CPUprepare-'+'-'.join(map(str,missing))+'.log'
   with (P/log).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-tilelang-original prepare tilelang-rms-expansion-cpu '+' '.join(map(str,missing)),stdout=out,stderr=subprocess.STDOUT,timeout=600).returncode
   assert rc==0,'CPUprepare failure retained; no blind libraryspin or deviceattempt'
  for i in ids:
   with (P/f'tilelang-rms-expansion-{i}-CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_tilelang_expansion_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'Cachedsource/input qualification failed'
  label='rollout-tilelang-rms-expansion-'+'-'.join(map(str,ids))+'-v1'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-tilelang-original','tilelang-rms-expansion',ids,label)
   if e is not None:break
   state.update(status='resource_deferred',checked_at=now());c.put(P/'TILELANG_RMS_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in e,'Guard/process failure retained, no blindretry'
  assert all(r.get('phase')!='load' for r in e['batch']['results']),'Commonloadfault stop affectedscope'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);attempted={r['ordinal'] for r in e['batch']['results']};rest=[i for i in ids if i not in attempted];assert attempted
  if rest:pending.insert(0,rest)
  state.update(last_batch=ids,accepted_original_APIs=report['accepted_original_APIs'],checked_at=now(),status='running');state.setdefault('failures',[]).extend({'ordinal':r['ordinal'],'error':r.get('error'),'checkpoint':label+'.CHECKPOINT.json'} for r in e['batch']['results'] if r['status']!='raw_checked_pending_guard');c.put(P/'TILELANG_RMS_EXPANSION.json',state)
 state['status']='all_exact_scope_attempted_preserve_failures'
except BaseException as ex:state.update(status='stopped_requires_diagnosis',error=repr(ex));raise
finally:
 state['checked_at']=now();c.put(P/'TILELANG_RMS_EXPANSION.json',state)
 if not previous_alive():c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
