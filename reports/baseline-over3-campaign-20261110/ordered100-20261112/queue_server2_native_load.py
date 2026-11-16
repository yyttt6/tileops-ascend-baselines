"""Private load qualification with original whole-task/fresh physical guard gates."""
import pathlib,ast,subprocess,datetime,json,fcntl,hashlib,shlex,os,time,sys
P=pathlib.Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';H='openlibing-DevEnv-055409';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';N=ROOT+'/native/ops-nn-v1';JOB=ROOT+'/job-python-v1/bin/python';GUARD='/usr/local/libexec/ascend-npu/npu_guard.py';known_active={}
sys.path.insert(0,str(P))
from nightly_window_gate import check as nightly_check
# Reuse the exact reviewed task observer definitions; do not execute its build main.
source=(P/'queue_server2_aligned_native_build.py').read_text();nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('now','put','remote','stopped','tasks')];assert len(nodes)==5;exec(compile(ast.Module(body=nodes,type_ignores=[]),'reviewed_server2_task_gate','exec'),globals())
lock=(P/'server2-native-load.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'entry':'queue_server2_native_load.py','status':'CPU_wait_for_fresh_admission','pid':os.getpid(),'host':H,'queued_at':now(),'no_baseline_or_numeric_acceptance':True};put(A/'SERVER2_NATIVE_LOAD.json',state)
try:
 build=json.loads((A/'SERVER2_REFERENCE_BUILD.json').read_text());assert build['status']=='built_pending_identity_and_load' and build['exit_code']==0 and build['guard']['performance_valid'] and build['guard']['workload_exit_code']==0
 m=json.loads((A/'ARTIFACT_MANIFEST.server2.json').read_text());assert not m['source_changes'];f=A/'load_native_package.py';dst=N+'/'+f.name;remote('python3 -c '+shlex.quote('import pathlib,sys;pathlib.Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())'),input=f.read_bytes(),check=True);assert remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(f.read_bytes()).hexdigest()
 while True:
  assert not stopped()
  nightly=nightly_check();state['nightly_window']=nightly
  if not nightly['allowed']:state.update(status='yield_0200_Shanghai_nightly_or_unobserved_dispatch',checked_at=now());put(A/'SERVER2_NATIVE_LOAD.json',state);time.sleep(60);continue
  okay,task=tasks();state.update(task_level=task,checked_at=now())
  if not okay:state['status']='yield_active_task_or_unknown';put(A/'SERVER2_NATIVE_LOAD.json',state);time.sleep(60);continue
  code='import pathlib,subprocess,shutil,json;print(json.dumps({"queue":[p.name for p in pathlib.Path("/run/lock/ascend-npu/device-0.queue").iterdir()],"workers":[s for s in subprocess.check_output(["ps","-eo","args"],text=True).splitlines() if s.split() and s.split()[0].endswith("Runner.Worker")],"disk":shutil.disk_usage("/").free}))';meta=json.loads(remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if meta['queue'] or meta['workers'] or meta['disk']<20000000000:state.update(status='yield_worker_queue_disk',meta=meta);put(A/'SERVER2_NATIVE_LOAD.json',state);time.sleep(30);continue
  r=remote('source '+ROOT+'/environment.sh; '+JOB+' '+GUARD+' status --json',stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
  if r.returncode:state.update(status='await_guard_idle',guard=r.stdout);put(A/'SERVER2_NATIVE_LOAD.json',state);time.sleep(30);continue
  g=json.loads(r.stdout);assert g['state']=='idle' and g['reservation']['locked'] is False
  okay,task=tasks()
  if not okay:continue
  nightly=nightly_check();state['nightly_window']=nightly
  if not nightly['allowed']:state.update(status='yield_0200_Shanghai_nightly_or_unobserved_dispatch',checked_at=now());put(A/'SERVER2_NATIVE_LOAD.json',state);time.sleep(60);continue
  tag='server2-native-load-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');owner='root/baseline-native-load/'+tag;receipt=N+'/'+tag+'-guard.json';result=tag+'.json';root=m['staging_root'];dirs=[root+'/op_api/lib',root+'/op_proto/es/lib/linux/aarch64',root+'/op_proto/lib/linux/aarch64',root+'/op_impl/ai_core/tbe/op_tiling'];env='env ASCEND_CUSTOM_OPP_PATH='+shlex.quote(root)+' LD_LIBRARY_PATH='+':'.join(dirs)+':"$LD_LIBRARY_PATH" '
  cmd='source '+ROOT+'/environment.sh; exec '+JOB+' '+GUARD+' run --owner '+owner+' --wait 10 --timeout 120 --receipt '+receipt+' -- '+env+JOB+' '+dst+' '+shlex.quote(owner)+' '+result
  put(A/(tag+'.GATE.json'),{'checked_at':now(),'tasks':task,'meta':meta,'guard':g});state.update(status='guarded_native_load',tag=tag,started_at=now());put(A/'SERVER2_NATIVE_LOAD.json',state)
  with (A/(tag+'.log')).open('w') as out:rc=remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
  got=remote('cat '+receipt,stdout=subprocess.PIPE,text=True);guard=json.loads(got.stdout) if got.returncode==0 else {};put(A/(tag+'-guard.json'),guard)
  if rc==75 and guard.get('phase')=='released_without_launch':state.update(status='deferred_no_child',checked_at=now());put(A/'SERVER2_NATIVE_LOAD.json',state);time.sleep(30);continue
  got=remote('cat '+N+'/'+result,stdout=subprocess.PIPE,text=True)
  data=json.loads(got.stdout) if got.returncode==0 else {};put(A/result,data);state.update(status='loaded_pending_numeric_qualification' if rc==0 and guard.get('performance_valid') else 'load_failed_requires_diagnosis',exit_code=rc,guard=guard,results=data.get('results',[]),result=result)
  for f in [A/(tag+'.log'),A/(tag+'-guard.json'),A/result]:
   target=N+'/'+f.name;code='import pathlib,sys;p=pathlib.Path('+repr(target)+');data=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==data;p.write_bytes(data)';remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);actual=remote('sha256sum '+target,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==hashlib.sha256(f.read_bytes()).hexdigest();state.setdefault('dual_archive',[]).append({'path':target,'sha256':actual})
  break
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:state['checked_at']=now();put(A/'SERVER2_NATIVE_LOAD.json',state)
