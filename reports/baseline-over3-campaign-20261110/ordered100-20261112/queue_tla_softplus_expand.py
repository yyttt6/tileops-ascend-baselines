"""Bounded exact original API cases after dynamic privatekernel qualification."""
import pathlib,json,datetime,fcntl,os,sys,subprocess,shlex,hashlib,time
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_tla_softplus_expand import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'tla-softplus-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'host':c.H,'queued_at':now(),'status':'source_deployment','scope':[2065,2066,2068],'accepted_original_APIs':0,'not_paired_speedup_or_public_update':True};c.put(P/'TLA_SOFTPLUS_EXPANSION.json',state)
try:
 assert not stop();rep=json.loads((P/'TLA_SOFTPLUS_CURRENT.json').read_text());assert rep['accepted_original_APIs']==1 and rep['rows'][0]['ordinal']==2067,'Softplus representative must fully qualify';prior=json.loads((P/'TLA_MEAN_CURRENT.json').read_text());assert prior['accepted_original_APIs']==1 and prior['rows'][0]['ordinal']==1968;terminal=json.loads((P/'rollout-tla-mean-1969-v4.FAILED.json').read_text());assert terminal['exit_code']==139 and terminal['guard']['phase']=='finished' and not terminal['guard']['performance_valid'];assert hashlib.sha256((P/(terminal['tag']+'.tar.gz')).read_bytes()).hexdigest()=='9c8ed7baf1f27d43be5b059b92759a9a7078ccf6a329f3287afc52675218ea87'
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  f=P/'rollout_bootstrap.py';s=f.read_text()
  if "'tla-softplus-expand':" not in s:
   f.with_name('rollout_bootstrap.before-tla-softplus.py').write_text(s);f.write_text(s.replace('script={',"script={'tla-softplus-expand':'worker_tla_softplus_expand.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','worker_tla_softplus_expand.py','tla_softplus_support.py','tla_hardsigmoid_support.py','SOFTPLUS_CPU_DOMAIN.json','softplus_cpu_domain.py','tla_softplus_expand_cached_fidelity.py','verify_tla_softplus_expand_cached.py']:
   f=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=h
  c.put(P/'TLA_SOFTPLUS_DEPLOYMENT.json',{'checked_at':now(),'files':records,'source_and_kernel_unchanged':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'selected_TLA_Softplus_three_complete_original_APIs_originalisolated15','durable_log':'tla-softplus-expand-queued.log','snapshot_read_only':True})
 for ids in [[2065,2066,2068]]:
  assert not stop()
  code='import pathlib,json;print(json.dumps([i for i in '+repr(ids)+' if not (pathlib.Path('+repr(c.R)+')/"prepared-rollout"/str(i)/"prepare.json").exists()]))'
  missing=json.loads(c.remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if missing:
   with (P/('tla-softplus-prepare-'+str(ids[0])+'.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-tla-original prepare tla-softplus-cpu '+' '.join(map(str,missing)),stdout=out,stderr=subprocess.STDOUT,timeout=600).returncode
   assert rc==0,'CPU-only prepare failure preserved'
  for i in ids:
   with (P/f'tla-softplus-{i}.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_tla_softplus_expand_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'Shared source/payload fault; do not loop failingcases'
  label='rollout-tla-softplus-'+'-'.join(map(str,ids))+'-v4'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-tla-original','tla-softplus-expand',ids,label)
   if e is not None:break
   state.update(status='resource_deferred',checked_at=now());c.put(P/'TLA_SOFTPLUS_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in e,'Guard/process fault archived; no blindretry'
  assert all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']),'Original checker/binding failure preserved; diagnose beforecontinuation'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=ids,accepted_original_APIs=report['accepted_original_APIs'],checked_at=now());c.put(P/'TLA_SOFTPLUS_EXPANSION.json',state)
 state['status']='representative_selected_three_API_route_complete'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=now();c.put(P/'TLA_SOFTPLUS_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
