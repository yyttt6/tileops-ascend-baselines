"""Finite original API representative then same-contract dtype expansion, no repeats."""
import pathlib,json,datetime,fcntl,os,sys,hashlib,subprocess,shlex,time
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_ops_nn_native92_all16 import summarize
stamp=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'ops-nn-native92-remaining-dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'queued_at':stamp(),'status':'CPU_deploy_and_validate','scope':[i for i in range(595,611) if i not in (602,606,608)],'protocol':'ascend950-isolated-ffts-v3','no_old_valid_measurement_repeated':True}
c.put(P/'OPS_NN_NATIVE92_REMAINING.json',state)
try:
 prior=json.loads((P/'OPS_NN_NATIVE92_EXPANSION.json').read_text());assert prior['status']=='all3_exact_original_APIs_fullchecker15raw_guard_valid' and prior['accepted_original_APIs']==3;assert not pathlib.Path('/proc/'+str(prior['pid'])+'/cmdline').exists()
 assert not stopped();q=json.loads((P/'server2-alignment-20261113/PRIVATE_DEVICE_QUALIFICATION92.json').read_text());assert q['private_original_device_kernels_verified'] is True
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  f=P/'rollout_bootstrap.py';old=f.read_text();assert "'ops-nn-native92-expand':" not in old;f.with_name('rollout_bootstrap.before-ops-nn-native92-expand.py').write_text(old);f.write_text(old.replace('script={',"script={'ops-nn-native92-expand':'worker_ops_nn_native92_expansion.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','worker_ops_nn_native92_expansion.py','ops_nn_foreach_api.py','ops_nn_foreach_mul_layout.py','ops_nn_mul_cached_fidelity.py','verify_ops_nn_mul_cached.py']:
   f=P/name;dst=c.R+'/'+name;code='import pathlib,sys;pathlib.Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=h
  c.put(P/'OPS_NN_NATIVE92_REMAINING_DEPLOYMENT.json',{'checked_at':stamp(),'files':records,'native_qualification':'server2-alignment-20261113/PRIVATE_DEVICE_QUALIFICATION92.json','CPUproof':'OPS_NN_MUL_PACK_CPU_PROOF.json','frozen_inputproof':'OPS_NN_MUL_NEXT_READY.json'})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':stamp(),'phase':'original_pinned_complete_native_Mul_originalisolated15_fullchecker','durable_log':'ops-nn-native92-remaining-queued.log','snapshot_read_only':True})
 scope=state['scope']
 query='import pathlib,json;r=pathlib.Path('+repr(c.R)+');ids='+repr(scope)+';print(json.dumps([i for i in ids if not (r/"prepared-rollout"/str(i)/"prepare.json").exists()]))'
 missing=json.loads(c.remote('python3 -c '+shlex.quote(query),stdout=subprocess.PIPE,text=True,check=True).stdout)
 state.update(status='CPU_prepare_missing_only',missing_inputs=missing,checked_at=stamp());c.put(P/'OPS_NN_NATIVE92_REMAINING.json',state)
 for begin in range(0,len(missing),4):
  ids=missing[begin:begin+4];assert not stopped()
  command='TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+c.R+'/prepare_rollout.py ops-nn-native92-remaining '+' '.join(map(str,ids))
  with (P/('ops-nn-native92-remaining-prepare-'+str(begin)+'.CPU.log')).open('w') as out:rc=c.remote(command,stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'CPU source/input preparation fault; no device import or blind repeatedscope'
 for ids in [scope[j:j+4] for j in range(0,len(scope),4)]:
  assert not stopped()
  for i in ids:
   with (P/f'ops-nn-native92-{i}.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_ops_nn_mul_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'Source/payload fault; stop sharedscope'
  label='rollout-ops-nn-native92-remaining-'+'-'.join(map(str,ids))+'-v1'
  while True:
   assert not stopped()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);event=c.run('r-template','ops-nn-native92-expand',ids,label)
   if event is not None:break
   state.update(status='resource_deferred',checked_at=stamp());c.put(P/'OPS_NN_NATIVE92_REMAINING.json',state);time.sleep(30)
  assert 'batch' in event,'Process/guard fault archived; no blindretry'
  for r in event['batch']['results']:assert r['status']=='raw_checked_pending_guard','Full originalchecker orbinding failed; preserve anddiagnose beforeexpanding'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=ids,accepted_original_APIs=report['accepted_original_APIs'],checked_at=stamp());c.put(P/'OPS_NN_NATIVE92_REMAINING.json',state)
  assert len(event['batch']['results'])==len(ids),'Do not silentlyskiprequested frozenkeys'
 state['status']='all13_remaining_exact_original_APIs_fullchecker15raw_guard_valid'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=stamp();c.put(P/'OPS_NN_NATIVE92_REMAINING.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':stamp(),'outcome':state['status'],'snapshot_read_only':True})
