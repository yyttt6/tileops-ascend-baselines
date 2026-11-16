"""Bounded exact original API cases after dynamic privatekernel qualification."""
import pathlib,json,datetime,fcntl,os,sys,subprocess,shlex,hashlib,time
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_ops_nn_unary import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'ops-nn-unary-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'host':c.H,'queued_at':now(),'status':'source_deployment','scope':list(range(385,390))+list(range(611,616)),'accepted_original_APIs':0,'not_paired_speedup_or_public_update':True};c.put(P/'OPS_NN_UNARY_EXPANSION.json',state)
try:
 assert not stop();q=json.loads((P/'server2-alignment-20261113/PRIVATE_UNARY_QUALIFICATION.json').read_text());assert q['private_original_device_kernels_verified'] is True
 prior=json.loads((P/'OPS_NN_NATIVE92_CURRENT.json').read_text());assert prior['accepted_original_APIs']==16
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  f=P/'rollout_bootstrap.py';s=f.read_text()
  if "'ops-nn-unary':" not in s:
   f.with_name('rollout_bootstrap.before-ops-nn-unary.py').write_text(s);f.write_text(s.replace('script={',"script={'ops-nn-unary':'worker_ops_nn_unary.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','worker_ops_nn_unary.py','ops_nn_unary_api.py','ops_nn_unary_cached_fidelity.py','verify_ops_nn_unary_cached.py']:
   f=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=h
  c.put(P/'OPS_NN_UNARY_DEPLOYMENT.json',{'checked_at':now(),'files':records,'qualification_sha256':hashlib.sha256((P/'server2-alignment-20261113/PRIVATE_UNARY_QUALIFICATION.json').read_bytes()).hexdigest(),'source_and_kernel_unchanged':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'pinned_complete_original_AbsNeg_fullchecker_originalisolated15','durable_log':'ops-nn-unary-queued.log','snapshot_read_only':True})
 for ids in [[385,386,387],[388],[389],[611,612,613],[614],[615]]:
  assert not stop()
  for i in ids:
   with (P/f'ops-nn-unary-{i}.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_ops_nn_unary_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'Shared source/payload fault; do not loop failingcases'
  label='rollout-ops-nn-unary-'+'-'.join(map(str,ids))+'-v1'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','ops-nn-unary',ids,label)
   if e is not None:break
   state.update(status='resource_deferred',checked_at=now());c.put(P/'OPS_NN_UNARY_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in e,'Guard/process fault archived; no blindretry'
  assert all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']),'Original checker/binding failure preserved; diagnose beforecontinuation'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=ids,accepted_original_APIs=report['accepted_original_APIs'],checked_at=now());c.put(P/'OPS_NN_UNARY_EXPANSION.json',state)
 state['status']='all10_exact_original_APIs_fullchecker15raw_guard_valid'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=now();c.put(P/'OPS_NN_UNARY_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
