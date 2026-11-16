"""Bounded exact original API cases after dynamic privatekernel qualification."""
import pathlib,json,datetime,fcntl,os,sys,subprocess,shlex,hashlib,time
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_tla_mean_expand import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'tla-mean-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'host':c.H,'queued_at':now(),'status':'source_deployment','scope':[1969],'accepted_original_APIs':0,'not_paired_speedup_or_public_update':True};c.put(P/'TLA_MEAN_EXPANSION.json',state)
try:
 assert not stop();rep=json.loads((P/'TLA_MEAN_CURRENT.json').read_text());assert rep['accepted_original_APIs']==1 and rep['rows'][0]['ordinal']==1968,'Mean representative must fully qualify';prior=json.loads((P/'TLA_LOGSUMEXP_CURRENT.json').read_text());assert prior['accepted_original_APIs']==3 and {r['ordinal'] for r in prior['rows']}=={1890,1891,1892},'EligibleLogSumExp3 must complete before Mean representative'
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  f=P/'rollout_bootstrap.py';s=f.read_text()
  if "'tla-mean-expand':" not in s:
   f.with_name('rollout_bootstrap.before-tla-mean.py').write_text(s);f.write_text(s.replace('script={',"script={'tla-mean-expand':'worker_tla_mean_expand.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','worker_tla_mean_expand.py','tla_mean_support.py','tla_hardsigmoid_support.py','tla_mean_expand_cached_fidelity.py','verify_tla_mean_expand_cached.py']:
   f=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=h
  c.put(P/'TLA_MEAN_DEPLOYMENT.json',{'checked_at':now(),'files':records,'source_and_kernel_unchanged':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'selected_TLA_Mean_two_complete_original_APIs_originalisolated15','durable_log':'tla-mean-expand-queued.log','snapshot_read_only':True})
 for ids in [[1969]]:
  assert not stop()
  code='import pathlib,json;print(json.dumps([i for i in '+repr(ids)+' if not (pathlib.Path('+repr(c.R)+')/"prepared-rollout"/str(i)/"prepare.json").exists()]))'
  missing=json.loads(c.remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if missing:
   with (P/('tla-mean-prepare-'+str(ids[0])+'.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-tla-original prepare tla-mean-cpu '+' '.join(map(str,missing)),stdout=out,stderr=subprocess.STDOUT,timeout=600).returncode
   assert rc==0,'CPU-only prepare failure preserved'
  for i in ids:
   with (P/f'tla-mean-{i}.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_tla_mean_expand_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'Shared source/payload fault; do not loop failingcases'
  label='rollout-tla-mean-'+'-'.join(map(str,ids))+'-v4'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-tla-original','tla-mean-expand',ids,label)
   if e is not None:break
   state.update(status='resource_deferred',checked_at=now());c.put(P/'TLA_MEAN_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in e,'Guard/process fault archived; no blindretry'
  assert all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']),'Original checker/binding failure preserved; diagnose beforecontinuation'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=ids,accepted_original_APIs=report['accepted_original_APIs'],checked_at=now());c.put(P/'TLA_MEAN_EXPANSION.json',state)
 state['status']='representative_selected_three_API_route_complete'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=now();c.put(P/'TLA_MEAN_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
