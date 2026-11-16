"""Single not-yet-attempted FP32 exactkey; no retry old256Mtimeout."""
from pathlib import Path
import json,datetime,fcntl,os,sys,subprocess,shlex,hashlib,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'ops-tensor-trunc-current.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'queued_at':now(),'status':'CPU_prepare','scope':[735],'accepted_original_APIs':0,'not_paired_or_public_update':True,'old256M_timeout_not_retried':True};c.put(P/'OPS_TENSOR_TRUNC_CURRENT_EXPANSION.json',state)
try:
 assert not stop();prior=json.loads((P/'TORCH_FUSEDQK_REMAINING_EXPANSION.json').read_text());assert prior['accepted_original_APIs']==3 and prior['status']=='all3_available_exact_original_Torch_FusedQK_fullchecker15raw_guard_valid_2catalog_unavailable'
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'single_original_opsTensor_Trunc16M_FP32_prepare_then_isolated15','durable_log':'ops-tensor-trunc-current-queued.log','snapshot_read_only':True})
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);f=P/'rollout_bootstrap.py';s=f.read_text()
  if "'ops-tensor-trunc-current':" not in s:f.with_name('rollout_bootstrap.before-ops-tensor-trunc-current.py').write_text(s);f.write_text(s.replace('script={',"script={'ops-tensor-trunc-current':'worker_ops_tensor_trunc_current.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','worker_ops_tensor_trunc_current.py']:
   f=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=h
  c.put(P/'OPS_TENSOR_TRUNC_CURRENT_DEPLOYMENT.json',{'checked_at':now(),'files':records,'provider_timedmath_kernel_checker_unchanged':True,'full_originalisolated15':True})
 with (P/'ops-tensor-trunc-735-prepare.CPU.log').open('w') as out:rc=c.remote('TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+c.R+'/prepare_rollout.py ops-tensor-trunc-current 735',stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,'Preserve CPU/source/case fault, no deviceattempt'
 label='rollout-ops-tensor-trunc-current-735-v1'
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-sdk91','ops-tensor-trunc-current',[735],label)
  if e is not None:break
  state.update(status='resource_deferred',checked_at=now());c.put(P/'OPS_TENSOR_TRUNC_CURRENT_EXPANSION.json',state);time.sleep(30)
 assert 'batch' in e,'Guard/process fault retained; no retryunchanged'
 assert all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']),'Originalfactory/checker failure retained'
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);subprocess.run([sys.executable,str(P/'summarize_ops_tensor_trunc_current.py')],check=True);subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='single_exact_originalAPI_fullchecker15raw_guard_valid',accepted_original_APIs=1)
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:
 state['checked_at']=now();c.put(P/'OPS_TENSOR_TRUNC_CURRENT_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
