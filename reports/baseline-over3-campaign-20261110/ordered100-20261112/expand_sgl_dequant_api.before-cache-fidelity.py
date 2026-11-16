"""Exact frozen per-tensor cases; retain original kernels/checker and both complete arms."""
import pathlib,sys,json,fcntl,time,subprocess,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_sgl_dequant_api import summarize
lock=(P/'sgl-dequant-api-expansion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert any(r['ordinal']==992 for r in summarize()['rows']),'representative must be audited'
indices=[991,990,988,987,986,989];state={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'indices':indices,'representative':992,'only_frozen_PerTensor_scope':True,'original_and_candidate_complete_API_each_15raw':True,'no_kernel_checker_precision_change':True,'no_predictive_memory_gate':True,'snapshot_read_only':True,'effective_retest_coverage_not_incremented_for_repeated_case':True}
c.put(P/'SGL_DEQUANT_API_EXPANSION.json',state)
for i in indices:
 if i in {r['ordinal'] for r in summarize()['rows']}:continue
 if (P/'STOP').exists() or (P.parent/'STOP').exists():raise RuntimeError('user STOP')
 labels=[f'rollout-sgl-dequant-api-candidate-{i}-original-v1',f'rollout-sgl-dequant-api-candidate-{i}-natural-v1']
 assert not any((P/(name+'.FAILED.json')).exists() for name in labels),'existing invalidguard attempt requires explicit evidence review/newcondition'
 with (P/(labels[0]+'.CPU.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-sgl prepare '+labels[0]+'-prepare '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,'CPUdependency orsource/payload identity failed; stop affectedgroup'
 for mode,label in zip(['auto','sgl-dequant-api'],labels):
  while True:
   with (P/'controller.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX);e=c.run('r-sgl',mode,[i],label)
   if e is not None:break
   time.sleep(30)
  assert 'batch' in e,'bootstrap/guard invalid; preserve evidence before recovery'
  if e['batch']['results'][0]['status']!='raw_checked_pending_guard':
   state.setdefault('case_failures',[]).append({'ordinal':i,'mode':mode,'checkpoint':label+'.CHECKPOINT.json'});c.put(P/'SGL_DEQUANT_API_EXPANSION.json',state);break
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
 r=summarize();state.update({'complete_pairs':r['complete_pairs'],'last_ordinal':i,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()});c.put(P/'SGL_DEQUANT_API_EXPANSION.json',state)
state['status']='all_exact_scope_attempted_preserve_failures';c.put(P/'SGL_DEQUANT_API_EXPANSION.json',state)
