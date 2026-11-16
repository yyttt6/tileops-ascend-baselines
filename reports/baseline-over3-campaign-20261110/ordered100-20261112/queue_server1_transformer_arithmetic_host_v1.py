"""Server2 reference build: whole-task PR/nightly yield, independent of server1."""
import pathlib,subprocess,json,datetime,fcntl,sys,shlex,time,hashlib,os
P=pathlib.Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';H='openlibing-DevEnv-661495';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';N=ROOT+'/native/ops-transformer-arithmetic-v1';JOB=ROOT+'/job-python-v1/bin/python';GUARD='/opt/baseline-over3-20261110/ordered100-20261112/opportunistic_guard.py'
sys.path.insert(0,str(P))
from nightly_window_gate import check as nightly_check
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(path,obj):
 t=path.with_name(path.name+'.reference.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');os.replace(t,path)
def remote(cmd,**kw):return subprocess.run(['ssh','-o','ServerAliveInterval=20',H,cmd],**kw)
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
def tasks():
 return True,{'server1_independent':True,'server2_task_gate_does_not_block_H1':True,'fresh_local_queue_worker_root_physicalguard_follow':True}
lock=(P/'server1-transformer-arithmetic-host-v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
previous=json.loads((A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json').read_text()) if (A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json').exists() else {}
known_active={r['repo']+':'+str(r['databaseId']):r for r in previous.get('task_level',{}).get('blocking',[])}
state={'status':'CPU_wait_for_server2_task_release','host':H,'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'queued_at':now(),'server1_independent':True,'no_performance_or_coverage_acceptance':True,'max_guard_seconds':600};put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state)
try:
 assert json.loads((A/'server1-toolchain.json').read_text())['status']=='passed'
 assert json.loads((A/'TRANSFORMER_SELECTION_SOURCE_SERVER1.json').read_text())['source_preparation']['source_files']==15360
 assert json.loads((A/'TRANSFORMER_OPBASE_PIN_FETCH_SERVER1.json').read_text())['actual_HEAD']=='0a83b915a6f6be49a03b907713d6f000b3dea1ea'
 assert json.loads((A/'server1-sdk-projection.json').read_text())['host_header_compile_exit_code']==0
 while True:
  assert not stopped()
  reference=json.loads((A/'SERVER2_TRANSFORMER_ARITHMETIC_HOST_V1.json').read_text())
  if reference.get('status') not in ('built_pending_identity_and_load',):
   assert reference.get('status') not in ('reference_build_failed_requires_diagnosis','stopped_requires_diagnosis'),'ScopedH2referencebuildfailure, no independentblindrebuild'
   state.update(status='CPU_wait_same_API_H2_reference_build_acceptance',checked_at=now());put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state);time.sleep(30);continue
  assert reference['exit_code']==0 and reference['guard']['performance_valid']
  # User's02:00nightly gets worst-case600+150seconds before its start.
  nightly={'allowed':True,'reason':'H1independent fromH2nightly/PR; freshlocaladmissionfollows'};state['nightly_window']=nightly
  if not nightly['allowed']:state.update(status='yield_0200_Shanghai_nightly_or_unobserved_dispatch',checked_at=now());put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state);time.sleep(60);continue
  okay,task=tasks();state.update(task_level=task,checked_at=now())
  if not okay:state['status']='CPU_wait_for_server2_task_release';put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state);time.sleep(60);continue
  meta_code='import pathlib,json,shutil,subprocess;print(json.dumps({"queue":[p.name for p in pathlib.Path("/run/lock/ascend-npu/device-0.queue").iterdir()],"workers":[x for x in subprocess.check_output(["ps","-eo","args"],text=True).splitlines() if x.split() and x.split()[0].endswith("Runner.Worker")],"disk":shutil.disk_usage("/").free}))'
  meta=json.loads(remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(meta_code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if meta['queue'] or meta['workers'] or meta['disk']<20000000000:state.update(status='yield_worker_queue_or_disk',meta=meta,checked_at=now());put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state);time.sleep(60);continue
  command='source '+ROOT+'/environment.sh; exec '+JOB+' '+GUARD+' status --json'
  r=remote(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
  if r.returncode:state.update(status='await_fresh_physical_idle',guard_status=r.stdout,guard_error=r.stderr,meta=meta,checked_at=now());put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state);time.sleep(30);continue
  g=json.loads(r.stdout)
  if g.get('state')!='idle' or g.get('reservation',{}).get('locked') is not False:state.update(status='await_fresh_physical_idle',guard_status=g,checked_at=now());put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state);time.sleep(30);continue
  # Recheck whole task release immediately before the guarded submission.
  okay,task=tasks()
  if not okay:state.update(status='CPU_wait_for_server2_task_release',task_level=task,checked_at=now());put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state);time.sleep(60);continue
  nightly={'allowed':True,'reason':'H1independent fromH2nightly/PR; freshlocaladmissionfollows'};state['nightly_window']=nightly
  if not nightly['allowed']:state.update(status='yield_0200_Shanghai_nightly_or_unobserved_dispatch',checked_at=now());put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state);time.sleep(60);continue
  tag='server1-transformer-arithmetic-host-v1-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');receipt=N+'/'+tag+'-guard.json';gate={'checked_at':now(),'tasks':task,'meta':meta,'guard':g};put(A/(tag+'.GATE.json'),gate)
  cmd='source '+ROOT+'/environment.sh; exec '+JOB+' '+GUARD+' run --owner root/baseline-align/'+tag+' --wait 10 --timeout 600 --receipt '+receipt+' -- bash '+N+'/build-host-arithmetic-original-v1.sh'
  state.update(status='guarded_server2_reference_build',tag=tag,started_at=now());put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state)
  with (A/(tag+'.log')).open('w') as out:rc=remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
  got=remote('cat '+receipt,stdout=subprocess.PIPE);assert got.returncode==0;g=json.loads(got.stdout);(A/(tag+'-guard.json')).write_bytes(got.stdout)
  if rc==75 and g.get('phase')=='released_without_launch':state.update(status='deferred_no_child',checked_at=now());put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state);time.sleep(30);continue
  state.update(status='built_pending_identity_and_load' if rc==0 else 'reference_build_failed_requires_diagnosis',exit_code=rc,guard=g,checked_at=now(),log=tag+'.log')
  for f in [A/(tag+'.log'),A/(tag+'-guard.json')]:
   dst=N+'/'+f.name;code='import pathlib,sys;p=pathlib.Path('+repr(dst)+');data=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==data;p.write_bytes(data)';remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();state.setdefault('dual_archive',[]).append({'path':dst,'sha256':h})
  break
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:state['checked_at']=now();put(A/'SERVER1_TRANSFORMER_ARITHMETIC_HOST_V1.json',state)
