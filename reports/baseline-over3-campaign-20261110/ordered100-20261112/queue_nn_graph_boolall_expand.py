"""Bounded exact original API cases after dynamic privatekernel qualification."""
import pathlib,json,datetime,fcntl,os,sys,subprocess,shlex,hashlib,time
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_nn_graph_boolall_expand import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'nn-graph-boolall-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'host':c.H,'queued_at':now(),'status':'source_deployment','scope':[406,407,409],'accepted_original_APIs':0,'not_paired_speedup_or_public_update':True};c.put(P/'NN_GRAPH_BOOLALL_EXPANSION.json',state)
try:
 assert not stop();predecessor=json.loads((P/'NN_GRAPH_MIN_CURRENT.json').read_text());assert predecessor['accepted_original_APIs']==5 and {r['ordinal'] for r in predecessor['rows']}=={410,411,412,413,414},'Amin5 must complete before BoolAll dispatch';prior=json.loads((P/'server2-alignment-20261113/SERVER1_GRAPH92_MIN_QUALIFICATION.json').read_text());assert prior['exit_code']==0 and prior['guard']['performance_valid']
 assert json.loads((P/'NN_GRAPH_BOOLALL_CURRENT.json').read_text())['accepted_original_APIs']==1,'408 representative must be valid'
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  f=P/'rollout_bootstrap.py';s=f.read_text()
  if "'nn-graph-boolall-expand':" not in s:
   f.with_name('rollout_bootstrap.before-nn-graph-boolall.py').write_text(s);f.write_text(s.replace('script={',"script={'nn-graph-boolall-expand':'worker_nn_graph_boolall_expand.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','worker_nn_graph_boolall_expand.py','nn_graph_boolall_support.py','nn_graph_boolall_expand_cached_fidelity.py','verify_nn_graph_boolall_expand_cached.py']:
   f=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=h
  c.put(P/'NN_GRAPH_BOOLALL_DEPLOYMENT.json',{'checked_at':now(),'files':records,'source_and_kernel_unchanged':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'original_GraphCall_BoolAll_repaired_profile_originalisolated15','durable_log':'nn-graph-boolall-expand-queued.log','snapshot_read_only':True})
 for ids in [[406,407,409]]:
  assert not stop()
  code='import pathlib,json;print(json.dumps([i for i in '+repr(ids)+' if not (pathlib.Path('+repr(c.R)+')/"prepared-rollout"/str(i)/"prepare.json").exists()]))'
  missing=json.loads(c.remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if missing:
   with (P/('nn-graph-boolall-prepare-'+str(ids[0])+'.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-template prepare nn-graph-boolall-cpu '+' '.join(map(str,missing)),stdout=out,stderr=subprocess.STDOUT,timeout=600).returncode
   assert rc==0,'CPU-only prepare failure preserved'
  for i in ids:
   with (P/f'nn-graph-boolall-{i}.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_nn_graph_boolall_expand_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'Shared source/payload fault; do not loop failingcases'
  label='rollout-nn-graph-boolall-'+'-'.join(map(str,ids))+'-v2'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','nn-graph-boolall-expand',ids,label)
   if e is not None:break
   state.update(status='resource_deferred',checked_at=now());c.put(P/'NN_GRAPH_BOOLALL_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in e,'Guard/process fault archived; no blindretry'
  assert all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']),'Original checker/binding failure preserved; diagnose beforecontinuation'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=ids,accepted_original_APIs=report['accepted_original_APIs'],checked_at=now());c.put(P/'NN_GRAPH_BOOLALL_EXPANSION.json',state)
 state['status']='all4_exact_original_APIs_fullchecker15raw_guard_valid'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=now();c.put(P/'NN_GRAPH_BOOLALL_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
