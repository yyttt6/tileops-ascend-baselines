"""Server2 reference build: whole-task PR/nightly yield, independent of server1."""
import pathlib,subprocess,json,datetime,fcntl,sys,shlex,time,hashlib,os
P=pathlib.Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';H='openlibing-DevEnv-661495';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';N=ROOT+'/native/ops-tensor-trunc-host-tile-v1';JOB=ROOT+'/job-python-v1/bin/python';GUARD='/opt/baseline-over3-20261110/ordered100-20261112/opportunistic_guard.py'
sys.path.insert(0,str(P))
from nightly_window_gate import check as nightly_check
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(path,obj):
 t=path.with_name(path.name+'.reference.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');os.replace(t,path)
def remote(cmd,**kw):return subprocess.run(['ssh','-o','ServerAliveInterval=20',H,cmd],**kw)
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
def tasks():return True,{'server1_independent':True,'server2_workflows_do_not_block_this_host':True,'local_worker_queue_physical_guard_checks_follow':True}

lock=(P/'server1-native-trunc-tile-probe-v2-build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
previous=json.loads((A/'SERVER1_TRUNC_TILE_PROBE_V2.json').read_text()) if (A/'SERVER1_TRUNC_TILE_PROBE_V2.json').exists() else {}
known_active={r['repo']+':'+str(r['databaseId']):r for r in previous.get('task_level',{}).get('blocking',[])}
state={'status':'CPU_wait_for_server2_task_release','host':H,'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'queued_at':now(),'server1_independent':True,'no_performance_or_coverage_acceptance':True,'max_guard_seconds':600};put(A/'SERVER1_TRUNC_TILE_PROBE_V2.json',state)
try:
 assert json.loads((A/'server1-toolchain.json').read_text())['status']=='passed'
 build=json.loads((A/'TRUNC_HOST_TILE_BUILD_HOST1.json').read_text());assert build['exit_code']==0 and build['host_wrapper_only_exposed_shape_arguments']['no_kernel_body_or_algorithm_edits']
 preflight='from pathlib import Path;import json,hashlib;p=Path('+repr(N)+');m=json.loads((p/"BUILD_RECEIPT.json").read_text());assert hashlib.sha256((p/"libops_tensor_trunc_tile.so").read_bytes()).hexdigest()==m["binary_sha256"];assert (p/"probe_trunc_host_tile.py").is_file() and (p/"CPU_BOUNDARY.json").is_file();print(json.dumps({"actual_N":str(p),"probe_sha256":hashlib.sha256((p/"probe_trunc_host_tile.py").read_bytes()).hexdigest(),"binary_sha256":m["binary_sha256"]}))'
 got=remote('python3 -I -S -c '+shlex.quote(preflight),stdout=subprocess.PIPE,check=True);pre=json.loads(got.stdout);assert pre['actual_N']==N;put(A/('TRUNC_TILE_ROUTE_PREFLIGHT_1.json'),pre)
 while True:
  assert not stopped()
  # User's02:00nightly gets worst-case600+150seconds before its start.
  nightly={'allowed':True,'reason':'server1 independent; server2 nightly task gate does not block H1'};state['nightly_window']=nightly
  if not nightly['allowed']:state.update(status='yield_0200_Shanghai_nightly_or_unobserved_dispatch',checked_at=now());put(A/'SERVER1_TRUNC_TILE_PROBE_V2.json',state);time.sleep(60);continue
  okay,task=tasks();state.update(task_level=task,checked_at=now())
  if not okay:state['status']='CPU_wait_for_server2_task_release';put(A/'SERVER1_TRUNC_TILE_PROBE_V2.json',state);time.sleep(60);continue
  meta_code='import pathlib,json,shutil,subprocess;print(json.dumps({"queue":[p.name for p in pathlib.Path("/run/lock/ascend-npu/device-0.queue").iterdir()],"workers":[x for x in subprocess.check_output(["ps","-eo","args"],text=True).splitlines() if x.split() and x.split()[0].endswith("Runner.Worker")],"disk":shutil.disk_usage("/").free}))'
  meta=json.loads(remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(meta_code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if meta['queue'] or meta['workers'] or meta['disk']<20000000000:state.update(status='yield_worker_queue_or_disk',meta=meta,checked_at=now());put(A/'SERVER1_TRUNC_TILE_PROBE_V2.json',state);time.sleep(60);continue
  command='source '+ROOT+'/environment.sh; exec '+JOB+' '+GUARD+' status --json'
  r=remote(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
  if r.returncode:state.update(status='await_fresh_physical_idle',guard_status=r.stdout,guard_error=r.stderr,meta=meta,checked_at=now());put(A/'SERVER1_TRUNC_TILE_PROBE_V2.json',state);time.sleep(30);continue
  g=json.loads(r.stdout)
  if g.get('state')!='idle' or g.get('reservation',{}).get('locked') is not False:state.update(status='await_fresh_physical_idle',guard_status=g,checked_at=now());put(A/'SERVER1_TRUNC_TILE_PROBE_V2.json',state);time.sleep(30);continue
  # Recheck whole task release immediately before the guarded submission.
  okay,task=tasks()
  if not okay:state.update(status='CPU_wait_for_server2_task_release',task_level=task,checked_at=now());put(A/'SERVER1_TRUNC_TILE_PROBE_V2.json',state);time.sleep(60);continue
  nightly={'allowed':True,'reason':'server1 independent; server2 nightly task gate does not block H1'};state['nightly_window']=nightly
  if not nightly['allowed']:state.update(status='yield_0200_Shanghai_nightly_or_unobserved_dispatch',checked_at=now());put(A/'SERVER1_TRUNC_TILE_PROBE_V2.json',state);time.sleep(60);continue
  tag='server1-native-trunc-tile-probe-v2-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');receipt=N+'/'+tag+'-guard.json';gate={'checked_at':now(),'tasks':task,'meta':meta,'guard':g};put(A/(tag+'.GATE.json'),gate)
  probe_result=tag+'.json';owner='root/baseline-native-probe/'+tag
  cmd='source '+ROOT+'/environment.sh; exec '+JOB+' '+GUARD+' run --owner '+owner+' --wait 10 --timeout 600 --receipt '+receipt+' -- '+JOB+' '+N+'/probe_trunc_host_tile.py '+owner+' '+probe_result
  state.update(status='guarded_server2_reference_build',tag=tag,started_at=now());put(A/'SERVER1_TRUNC_TILE_PROBE_V2.json',state)
  with (A/(tag+'.log')).open('w') as out:rc=remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
  got=remote('cat '+receipt,stdout=subprocess.PIPE);assert got.returncode==0;g=json.loads(got.stdout);(A/(tag+'-guard.json')).write_bytes(got.stdout)
  if rc==75 and g.get('phase')=='released_without_launch':state.update(status='deferred_no_child',checked_at=now());put(A/'SERVER1_TRUNC_TILE_PROBE_V2.json',state);time.sleep(30);continue
  result_data=capture_result=None
  got_result=remote('cat '+N+'/'+probe_result,stdout=subprocess.PIPE);assert got_result.returncode==0;result_data=json.loads(got_result.stdout);(A/probe_result).write_bytes(got_result.stdout)
  state.update(result=probe_result,results=result_data.get('rows',[]),status='functional_API_passed_private_attribution_pending' if rc==0 else 'functional_probe_failed_requires_diagnosis',exit_code=rc,guard=g,checked_at=now(),log=tag+'.log')
  for f in [A/(tag+'.log'),A/(tag+'-guard.json'),A/probe_result]:
   dst=N+'/'+f.name;code='import pathlib,sys;p=pathlib.Path('+repr(dst)+');data=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==data;p.write_bytes(data)';remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();state.setdefault('dual_archive',[]).append({'path':dst,'sha256':h})
  break
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:state['checked_at']=now();put(A/'SERVER1_TRUNC_TILE_PROBE_V2.json',state)
