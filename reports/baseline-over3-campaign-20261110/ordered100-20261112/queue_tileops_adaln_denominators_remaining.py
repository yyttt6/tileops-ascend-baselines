"""Only the remaining five unchanged public TileOps denominators, exact native-pair inputs."""
from pathlib import Path
import os,sys,json,datetime,fcntl,subprocess,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=P/'TILEOPS_ADALN_DENOMINATORS_REMAINING.json';lock=(P/'tileops-adaln-denominators-remaining.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'source_identity_preflight','scope':[196,197,198,199,200],'coverage_increment':0,'qualified_denominators':[],'queued_at':now()}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
put()
try:
 q=json.loads((P/'server2-alignment-20261113/TILEOPS_ADALN_DENOMINATOR_195_IDENTITY.json').read_text());assert q['full_original_public_Op_kernel_generated_binary_checker_raw_guard_identity_verified'];done=json.loads((P/'FINAL_SAME_ENV_CLOSURES_CURRENT.json').read_text());assert any(r['ordinal']==195 and r['final_under3_qualified'] for r in done['rows'])
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'Remaining5_unchanged_TileOps_AdaLN_denominators_after_first_qualified_under3','snapshot_read_only':True})
 for i in s['scope']:
  assert not stop();label=f'rollout-tileops-adaln-denominator-{i}-v3'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','tileops-adaln-denominator-v3',[i],label)
   if e is not None:break
   s.update(status='resource_deferred',current_ordinal=i);put();time.sleep(30)
  assert 'batch' in e,'Common guard/process fault preserved; no blind case-loop'
  r=e['batch']['results'][0]
  if r['status']!='raw_checked_pending_guard':
   c.put(P/f'TILEOPS_ADALN_DENOMINATOR_{i}_HOLD.json',{'ordinal':i,'checkpoint':label+'.CHECKPOINT.json','result':r,'no_samecondition_retry':True})
   if r['phase']=='measurement_and_original_verification' and 'not close' in r.get('error',''):s.setdefault('local_numerical_holds',[]).append(i);put();continue
   raise RuntimeError(('Common denominator dependency fault, do not cycle othercases',i,r))
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);subprocess.run([sys.executable,str(P/'audit_tileops_adaln_denominator_remaining.py'),str(i)],check=True);s['qualified_denominators'].append(i);s.update(status='running',current_ordinal=i);put()
  if (P/'summarize_adaln_final_all.py').exists():subprocess.run([sys.executable,str(P/'summarize_adaln_final_all.py')],check=True)
 s['status']='remaining_denominators_terminal_final_ratio_audit_required'
except BaseException as ex:s.update(status='stopped_requires_diagnosis',error=repr(ex));raise
finally:put();c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
