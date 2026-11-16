"""Whole-foreground H1 independent qualification of the exact H2 Graph capsule."""
from pathlib import Path
import json,subprocess,fcntl,os,sys,datetime,time,shlex,hashlib
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';sys.path.insert(0,str(P));import rollout_controller as c
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-vectorinf-v1';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';JOB=ROOT+'/job-python-v1/bin/python';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(f,v):
 t=f.with_suffix('.graph.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(f)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'server1-vectorinf-qualification.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s={'entry':Path(__file__).name,'pid':os.getpid(),'host':c.H,'queued_at':now(),'status':'CPU_wait_for_capsule_and_predecessor','no_formal_baseline_acceptance':True};file=A/'SERVER1_VECTORINF_QUALIFICATION.json';put(file,s)
try:
 while True:
  assert not stop()
  prior=json.loads((P/'TLA_VECTORINF_CURRENT.json').read_text()) if (P/'TLA_VECTORINF_CURRENT.json').exists() else {}
  if json.loads((P/'OPS_MATH_COUNTNZ_CURRENT.json').read_text())['accepted_original_APIs']==4:break
  s.update(status='CPU_wait_predecessor_CountNZ5_no_driver_probe',checked_at=now());put(file,s);time.sleep(5)
 assert json.loads((A/'VECTORINF_H2_NATIVE_QUALIFICATION.json').read_text())['private_original_device_kernels_verified'] is True
 prior=json.loads((A/'SERVER2_OPSMATH_VECTORINF_QUALIFICATION.json').read_text());assert prior['exit_code']==0 and prior['guard']['performance_valid']
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);okay,gate=c.admit()
   if okay:
    tag='server1-vectorinf-qualification-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');owner='root/baseline-native-probe/'+tag;receipt=N+'/'+tag+'-guard.json';result=tag+'.json';put(A/(tag+'.GATE.json'),gate);s.update(status='guarded_native_load',tag=tag,checked_at=now());put(file,s)
    command='source '+ROOT+'/environment.sh; exec '+JOB+' '+c.R+'/opportunistic_guard.py run --owner '+owner+' --wait 10 --timeout 120 --receipt '+receipt+' -- bash '+N+'/probe-entry.sh '+owner+' '+result
    with (A/(tag+'.log')).open('w') as out:rc=c.remote(command,stdout=out,stderr=subprocess.STDOUT).returncode
    raw=c.remote('cat '+receipt,stdout=subprocess.PIPE,check=True).stdout;guard=json.loads(raw);(A/(tag+'-guard.json')).write_bytes(raw)
   else:s.update(status='resource_deferred',checked_at=now());put(file,s)
  if not okay:time.sleep(30);continue
  if rc==75 and guard.get('phase')=='released_without_launch':time.sleep(30);continue
  raw=c.remote('cat '+N+'/'+result,stdout=subprocess.PIPE,check=True).stdout;(A/result).write_bytes(raw);data=json.loads(raw);passed=rc==0 and guard.get('performance_valid') is True and data.get('full_output_and_inputs_verified') is True;s.update(status='loaded_pending_numeric_qualification' if passed else 'load_failed_requires_diagnosis',exit_code=rc,guard=guard,result=result,diagnostic_result={k:data.get(k) for k in ['status','error','full_output_and_inputs_verified','runtime']})
  for suffix in ['.maps.txt','.pt']:
   path=N+'/'+result+suffix
   if c.remote('test -f '+path).returncode==0:
    raw=c.remote('cat '+path,stdout=subprocess.PIPE,check=True).stdout;(A/(result+suffix)).write_bytes(raw);assert c.remote('sha256sum '+path,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(raw).hexdigest()
  for f in [A/(tag+'.log'),A/(tag+'-guard.json'),A/result]:
   dest=N+'/'+f.name;code='from pathlib import Path;import sys;p=Path('+repr(dest)+');raw=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw)';c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=f.read_bytes(),check=True);assert c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(f.read_bytes()).hexdigest()
  break
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:s['checked_at']=now();put(file,s)
