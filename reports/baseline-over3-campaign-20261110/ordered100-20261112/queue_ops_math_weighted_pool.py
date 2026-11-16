"""Bounded exact original API cases after dynamic privatekernel qualification."""
import pathlib,json,datetime,fcntl,os,sys,subprocess,shlex,hashlib,time
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_ops_math_weighted_pool import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'ops-math-weighted-pool-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
state={'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'host':c.H,'queued_at':now(),'status':'source_deployment','scope':[220,226],'accepted_complete_APIs':0,'not_paired_speedup_or_public_update':True};c.put(P/'OPS_MATH_WEIGHTEDPOOL_EXPANSION.json',state)
try:
 assert not stop();prior=json.loads((P/'server2-alignment-20261113/MATH_WEIGHTED_POOL_H1_HOST_QUALIFICATION.json').read_text());assert prior['full_API_and_original_flush_and_private_kernel_verified'] is True
 for name,h in prior['qualified_API_source_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  f=P/'rollout_bootstrap.py';s=f.read_text()
  if "'ops-math-weighted-pool-v1':" not in s:
   f.with_name('rollout_bootstrap.before-ops-math-weighted-pool-v1.py').write_text(s);f.write_text(s.replace('script={',"script={'ops-math-weighted-pool-v1':'worker_ops_math_weighted_pool.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','worker_ops_math_weighted_pool.py','ops_math_weighted_pool_support.py','ops_math_weighted_pool_api.py','ops_math_realdiv_capi_v2.py','ops_math_l2_api.py','ops_math_sqrt_api.py','ops_math_l1_api.py','ops_math_dequant_api.py','ops_math_weighted_pool_cached_fidelity.py','verify_ops_math_weighted_pool_cached.py']:
   f=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=h
  c.put(P/'OPS_MATH_WEIGHTEDPOOL_DEPLOYMENT.json',{'checked_at':now(),'files':records,'source_and_kernel_unchanged':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'MathWeightedPool2_completeMulDivSum_fulloriginalwindowAST_originalisolated15','durable_log':'qualified-coverage-events.jsonl','snapshot_read_only':True})
 for ids in [[220],[226]]:
  assert not stop()
  accepted=set(json.loads((P/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100']);ids=[i for i in ids if i not in accepted]
  holds=json.loads((P/'MATH_WEIGHTED_POOL_REQUIRES_DIAGNOSIS.json').read_text()).get('ordinals',[]) if (P/'MATH_WEIGHTED_POOL_REQUIRES_DIAGNOSIS.json').exists() else [];assert not set(ids)&set(holds),'Failure hold requires diagnosis, no crosshost blindretry'
  if not ids:continue
  code='import pathlib,json;print(json.dumps([i for i in '+repr(ids)+' if not (pathlib.Path('+repr(c.R)+')/"prepared-rollout"/str(i)/"prepare.json").exists()]))'
  missing=json.loads(c.remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if missing:
   with (P/('ops-math-weighted-pool-prepare-'+str(ids[0])+'.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-template prepare ops-math-weighted-pool-cpu '+' '.join(map(str,missing)),stdout=out,stderr=subprocess.STDOUT,timeout=600).returncode
   assert rc==0,'CPU-only prepare failure preserved'
  for i in ids:
   with (P/f'ops-math-weighted-pool-{i}.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_ops_math_weighted_pool_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'Shared source/payload fault; do not loop failingcases'
  label='rollout-ops-math-weighted-pool-'+'-'.join(map(str,ids))+'-l1-v1'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','ops-math-weighted-pool-v1',ids,label)
   if e is not None:break
   state.update(status='resource_deferred',checked_at=now());c.put(P/'OPS_MATH_WEIGHTEDPOOL_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in e,'Guard/process fault archived; no blindretry'
  bad=[x['ordinal'] for x in e['batch']['results'] if x['status']!='raw_checked_pending_guard']
  if bad:c.put(P/'MATH_WEIGHTED_POOL_REQUIRES_DIAGNOSIS.json',{'ordinals':bad,'checkpoint':label+'.CHECKPOINT.json','no_samecondition_retry':True})
  assert all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']),'Original checker/binding failure preserved; diagnose beforecontinuation'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=ids,accepted_complete_APIs=report['accepted_complete_APIs'],checked_at=now());c.put(P/'OPS_MATH_WEIGHTEDPOOL_EXPANSION.json',state)
 state['status']='all2_complete_same_library_MeanPool_APIs_fullchecker15raw_guard_valid'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=now();c.put(P/'OPS_MATH_WEIGHTEDPOOL_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
