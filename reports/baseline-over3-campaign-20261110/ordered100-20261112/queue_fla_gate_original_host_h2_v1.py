"""One bounded original gate dependency build, independent per host admission."""
from pathlib import Path
import sys,json,datetime,os,fcntl,time,subprocess,hashlib,shlex
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
from nightly_window_gate import check as nightly_check
from server2_whole_task_submission_gate import tasks
H1='openlibing-DevEnv-055409';H2='openlibing-DevEnv-055409';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-gate-v1';STACK='/opt/baseline-over3-20261110/server2-aligned-stack';JOB=STACK+'/job-python-v1/bin/python';GUARD=c.R+'/opportunistic_guard.py';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=A/'FLA_GATE_ORIGINAL_HOST_H2_V1.json';state={'pid':os.getpid(),'entry':Path(__file__).name,'status':'await_fresh_per_host_build_admission','queued_at':now(),'not_coverage':True,'max_guard_seconds':600}
def put():state['checked_at']=now();c.put(f,state)
def stop():return (D/'STOP').exists() or (D.parent/'STOP').exists() or json.loads((D.parent/'STATE.json').read_text())['status']!='active'
def rpc(h,cmd,**kw):return subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8',h,cmd],**kw)
with (D/'fla-gate-build-H2.lock').open('a') as own:
 fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not f.exists(),'Existing build outcome must be diagnosed, not repeated';put()
 try:
  for k in ['H1','H2']:assert json.loads((A/('FLA_GATE_ORIGINAL_SOURCE_'+k+'_V1.json')).read_text())['source_files']==5479
  while True:
   assert not stop();chosen=None;gates={}
   with (D/'fla-gate-H2-private-build-slot.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);assert not stop()
    c.H=H1;okay,g=c.admit();gates['H1']=g
    if okay:chosen=H1
    else:
     gates['H2_nightly']=nightly_check()
     if gates['H2_nightly']['allowed']:
      released,t=tasks();gates['H2_tasks']=t
      if released:
       c.H=H2;okay,g=c.admit();gates['H2']=g
       if okay:chosen=H2
    if chosen==H2:
     released,t=tasks();gates['H2_fresh_tasks']=t;gates['H2_fresh_nightly']=nightly_check()
     if not released or not gates['H2_fresh_nightly']['allowed']:chosen=None
    if chosen is not None:
     assert not stop();tag='fla-gate-original-aot-h2-v1-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');receipt=N+'/'+tag+'-guard.json';c.put(A/(tag+'.GATE.json'),{'checked_at':now(),'gates':gates,'host':chosen});state.update(status='guarded_original_gate_dependency_build',host=chosen,tag=tag,gates=gates);put()
     cmd='source '+STACK+'/environment.sh; exec '+JOB+' '+GUARD+' run --owner root/baseline-align/'+tag+' --wait 10 --timeout 600 --receipt '+receipt+' -- bash '+N+'/build-host-fla-gate-v1.sh'
     with (A/(tag+'.log')).open('w') as out:rc=rpc(chosen,cmd,stdout=out,stderr=subprocess.STDOUT).returncode
     got=rpc(chosen,'cat '+receipt,stdout=subprocess.PIPE,check=True);g=json.loads(got.stdout);(A/(tag+'-guard.json')).write_bytes(got.stdout)
     if rc==75 and g.get('phase')=='released_without_launch':state.update(status='deferred_no_child',guard=g);put()
     else:
      state.update(exit_code=rc,guard=g,status='built_pending_source_ELF_identity_and_full_measurement' if rc==0 and g.get('performance_valid') and g.get('workload_exit_code')==0 else 'failed_requires_diagnosis_no_retry')
      for p in [A/(tag+'.log'),A/(tag+'-guard.json')]:
       raw=p.read_bytes();dst=N+'/'+p.name;code='from pathlib import Path;import sys,hashlib;p=Path('+repr(dst)+');raw=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw);print(hashlib.sha256(raw).hexdigest())';r=rpc(chosen,'python3 -I -S -c '+shlex.quote(code),input=raw,stdout=subprocess.PIPE,check=True);digest=hashlib.sha256(raw).hexdigest();assert r.stdout.decode().strip()==digest;state.setdefault('dual_archive',[]).append({'path':dst,'sha256':digest})
      if state['status']=='built_pending_source_ELF_identity_and_full_measurement':
       code='from pathlib import Path;import json,hashlib;p=Path('+repr(N)+');m=json.loads((p/"PACKAGE_AOT_V1_MANIFEST.json").read_text());assert all(hashlib.sha256((Path(m["package"])/n).read_bytes()).hexdigest()==h for n,h in m["artifact_sha256"].items());print(json.dumps(m))'
       got=rpc(chosen,'python3 -I -S -c '+shlex.quote(code),stdout=subprocess.PIPE,check=True);m=json.loads(got.stdout);c.put(A/'FLA_GATE_AOT_PACKAGE_H2.json',m);first=json.loads((A/'FLA_GATE_AOT_PACKAGE_H1.json').read_text());assert m['artifact_sha256']==first['artifact_sha256'],'Original host/device package SHA mismatch'
       c.put(D/'FLA_GATE_API_H2_READY.json',{'checked_at':now(),'ready':True,'reason':'Original5479 source and guarded950 gate build verified;341 package files rehashed, all artifacts equal H1; whole PR/nightly release and fresh physical admission still required'})
      put();break
   if chosen is None:state.update(status='await_fresh_per_host_build_admission',gates=gates);put()
   time.sleep(30)
 except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));put();raise
