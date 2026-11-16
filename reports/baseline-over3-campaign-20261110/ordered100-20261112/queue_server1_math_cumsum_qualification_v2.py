"""Whole-foreground H1 independent qualification of the exact H2 Graph capsule."""
from pathlib import Path
import json,subprocess,fcntl,os,sys,datetime,time,shlex,hashlib
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';sys.path.insert(0,str(P));import rollout_controller as c
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-cumscan-v1';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';JOB=ROOT+'/job-python-v1/bin/python';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(f,v):
 t=f.with_suffix('.graph.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(f)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'server1-math-cumscan-qualification-v2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s={'entry':Path(__file__).name,'pid':os.getpid(),'host':c.H,'queued_at':now(),'status':'CPU_wait_for_capsule_and_predecessor','no_formal_baseline_acceptance':True};file=A/'SERVER1_MATH_CUMSUM_QUALIFICATION_V2.json';put(file,s)
try:
 while True:
  assert not stop();stage=json.loads((A/'MATH_CUMSCAN_POSTCOMPILE_V4.json').read_text())
  if stage['status']=='stopped_requires_diagnosis':raise RuntimeError('Prior compile/package failure; no probe: '+stage.get('error',''))
  if stage['status']=='package_ready_full_API_device_qualification_required':break
  os.kill(stage['pid'],0);time.sleep(10)
 manifest=json.loads((A/'MATH_CUMSCAN_PACKAGE.json').read_text());assert manifest['original_source_unchanged'] and not manifest['runtime_qualified'];assert all(name in manifest['API_exports'] for name in ['aclnnCumsumGetWorkspaceSize','aclnnCumsum'])
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);okay,gate=c.admit()
   if okay:
    tag='server1-math-cumscan-qualification-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');owner='root/baseline-native-probe/'+tag;receipt=N+'/'+tag+'-guard.json';result=tag+'.json';put(A/(tag+'.GATE.json'),gate);s.update(status='guarded_native_load',tag=tag,checked_at=now());put(file,s)
    command='source '+ROOT+'/environment.sh; exec '+JOB+' '+c.R+'/opportunistic_guard.py run --owner '+owner+' --wait 10 --timeout 120 --receipt '+receipt+' -- bash '+N+'/probe-entry.sh '+owner+' '+result
    with (A/(tag+'.log')).open('w') as out:rc=c.remote(command,stdout=out,stderr=subprocess.STDOUT).returncode
    raw=c.remote('cat '+receipt,stdout=subprocess.PIPE,check=True).stdout;guard=json.loads(raw);(A/(tag+'-guard.json')).write_bytes(raw)
   else:s.update(status='resource_deferred',checked_at=now());put(file,s)
  if not okay:time.sleep(30);continue
  if rc==75 and guard.get('phase')=='released_without_launch':time.sleep(30);continue
  raw=c.remote('cat '+N+'/'+result,stdout=subprocess.PIPE,check=True).stdout;(A/result).write_bytes(raw);data=json.loads(raw);passed=rc==0 and guard.get('performance_valid') is True and data.get('full_output_and_inputs_verified') is True;s.update(status='full_API_flush_probe_passed_private_kernel_path_audit_required' if passed else 'load_failed_requires_diagnosis',exit_code=rc,guard=guard,result=result,diagnostic_result={k:data.get(k) for k in ['status','error','full_output_and_inputs_verified','runtime']})
  for suffix in ['.maps.txt','.pt']:
   path=N+'/'+result+suffix
   if c.remote('test -f '+path).returncode==0:
    raw=c.remote('cat '+path,stdout=subprocess.PIPE,check=True).stdout;(A/(result+suffix)).write_bytes(raw);assert c.remote('sha256sum '+path,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(raw).hexdigest()
  for f in [A/(tag+'.log'),A/(tag+'-guard.json'),A/result]:
   dest=N+'/'+f.name;code='from pathlib import Path;import sys;p=Path('+repr(dest)+');raw=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw)';c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=f.read_bytes(),check=True);assert c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(f.read_bytes()).hexdigest()
  break
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:s['checked_at']=now();put(file,s)
