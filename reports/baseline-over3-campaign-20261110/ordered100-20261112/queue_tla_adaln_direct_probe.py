from pathlib import Path
import json,sys,os,datetime,time,fcntl,subprocess,shlex,hashlib,tarfile
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
state={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_source_deployment','host':c.H,'queued_at':now(),'not_formal_coverage':True};record=P/'TLA_DIRECT_NORM_PROBE.json'
lock=(P/'tla-direct-norm-probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 failed=json.loads((P/'rollout-tla-adaln-f32-1614-v3.FAILED.json').read_text());assert failed['archive_sha256']=='079042c09ef1fd556ef0f175a2a1ef3e4dee37d5e36a465e42bdd9fae1111bde' and failed['guard']['phase']=='finished'
 c.put(record,state)
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);p=P/'rollout_bootstrap.py';s=p.read_text()
  if "'tla-direct-norm-probe':" not in s:p.with_name('rollout_bootstrap.before-direct-norm-probe.py').write_text(s);p.write_text(s.replace('script={',"script={'tla-direct-norm-probe':'worker_tla_adaln_direct_probe.py',",1))
  for name in ['rollout_bootstrap.py','tla_direct_norm_selector.py','worker_tla_adaln_direct_probe.py']:
   f=P/name;c.remote('cat > '+c.R+'/'+name,input=f.read_bytes(),check=True);assert c.remote('sha256sum '+c.R+'/'+name,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(f.read_bytes()).hexdigest()
 while True:
  assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);okay,gate=c.admit()
   if okay:
    tag='tla-direct-norm-probe-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');receipt=c.R+'/'+tag+'-guard.json';state.update(status='guarded_functional_probe',tag=tag,checked_at=now());c.put(record,state);c.put(P/(tag+'.GATE.json'),gate)
    cmd='source /opt/baseline-over3-20261110/unified-stack-v1/environment.sh; exec /opt/tileops-ci/runtime/bin/python -I -S '+c.R+'/opportunistic_guard.py run --owner root/baseline-direct-norm/'+tag+' --wait 10 --timeout 180 --receipt '+receipt+' -- bash '+c.R+'/launch.sh r-tla-original tla-direct-norm-probe '+tag
    with (P/(tag+'.log')).open('w') as out:rc=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
    guard=json.loads(c.remote('cat '+receipt,stdout=subprocess.PIPE,text=True,check=True).stdout);c.put(P/(tag+'-guard.json'),guard)
   else:state.update(status='resource_deferred',checked_at=now());c.put(record,state)
  if not okay:time.sleep(30);continue
  if rc==75 and guard.get('phase')=='released_without_launch':time.sleep(30);continue
  state.update(exit_code=rc,guard=guard,status='finished_pending_result_audit' if rc==0 else 'failed_requires_diagnosis')
  for name in [tag+'.log',tag+'-guard.json']:
   f=P/name;code='from pathlib import Path;import sys;p=Path('+repr(c.R+'/'+name)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b)';c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=f.read_bytes(),check=True);assert c.remote('sha256sum '+c.R+'/'+name,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(f.read_bytes()).hexdigest()
  if c.remote('test -f '+c.R+'/direct-norm-probes/'+tag+'/RESULT.json').returncode==0:
   raw=c.remote('cat '+c.R+'/direct-norm-probes/'+tag+'/RESULT.json',stdout=subprocess.PIPE,check=True).stdout;(P/(tag+'.RESULT.json')).write_bytes(raw);state.update(result=tag+'.RESULT.json',result_sha256=hashlib.sha256(raw).hexdigest());r=json.loads(raw);state['status']='direct_API_qualified_no_formal_measurement' if rc==0 and guard.get('performance_valid') and guard.get('workload_exit_code')==0 and r['status']=='direct_original_API_full_output_and_input_passed' else 'failed_requires_diagnosis'
  break
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:state['checked_at']=now();c.put(record,state)
