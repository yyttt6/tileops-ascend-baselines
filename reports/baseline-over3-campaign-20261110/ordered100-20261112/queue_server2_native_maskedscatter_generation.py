"""Server2 reference build: whole-task PR/nightly yield, independent of server1."""
import pathlib,subprocess,json,datetime,fcntl,sys,shlex,time,hashlib,os
P=pathlib.Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';H='openlibing-DevEnv-055409';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';N=ROOT+'/native/ops-nn-unary-maskedscatter-v1';JOB=ROOT+'/job-python-v1/bin/python';GUARD='/usr/local/libexec/ascend-npu/npu_guard.py'
sys.path.insert(0,str(P))
from nightly_window_gate import check as nightly_check
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(path,obj):
 t=path.with_name(path.name+'.reference.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');os.replace(t,path)
def remote(cmd,**kw):return subprocess.run(['ssh','-o','ServerAliveInterval=20',H,cmd],**kw)
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
def tasks():
 global known_active
 result={};blocking=[]
 # A long active workflow must not disappear behind a limit of newer completed runs.
 for key,row in list(known_active.items()):
  repo=row['repo'];r=subprocess.run(['gh','run','view',str(row['databaseId']),'-R',repo,'--json','databaseId,status,conclusion,workflowName,createdAt'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
  if r.returncode:return False,{'query_failed':repo,'run':row['databaseId'],'error':r.stderr}
  current=json.loads(r.stdout)
  if current['status']=='completed':known_active.pop(key)
  else:blocking.append({'repo':repo,**current})
 if blocking:return False,{'blocking':blocking,'source':'explicit previouslyobservedrunIDs, independent of recenthistorylimit'}
 for repo in ['yyttt6/TileOPs','yyttt6/TileOPs.github.io']:
  rows=[]
  for status in ['in_progress','queued','waiting','requested','action_required']:
   r=subprocess.run(['gh','run','list','-R',repo,'--status',status,'--limit','100','--json','databaseId,status,conclusion,workflowName,createdAt'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
   if r.returncode:return False,{'query_failed':repo,'status':status,'error':r.stderr}
   rows.extend(json.loads(r.stdout))
  result[repo]=rows
  for row in rows:
   name=row['workflowName'].lower();relevant=('ascend 950' in name) if repo.endswith('/TileOPs') else ('nightly' in name)
   if relevant and row['status']!='completed':
    value={'repo':repo,**row};blocking.append(value);known_active[repo+':'+str(row['databaseId'])]=value
 return not blocking,{'blocking':blocking,'runs':result}
lock=(P/'server2-native-maskedscatter-device-build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
previous=json.loads((A/'SERVER2_MASKEDSCATTER_GENERATION.json').read_text()) if (A/'SERVER2_MASKEDSCATTER_GENERATION.json').exists() else {}
known_active={r['repo']+':'+str(r['databaseId']):r for r in previous.get('task_level',{}).get('blocking',[])}
state={'status':'CPU_wait_for_server2_task_release','host':H,'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'queued_at':now(),'server1_independent':True,'no_performance_or_coverage_acceptance':True,'max_guard_seconds':600};put(A/'SERVER2_MASKEDSCATTER_GENERATION.json',state)
try:
 assert json.loads((A/'server2-toolchain.json').read_text())['status']=='passed'
 assert json.loads((A/'server2-maskedscatter-source.json').read_text())['source_files_verified']==25610
 assert json.loads((A/'server2-sdk-projection.json').read_text())['host_header_compile_exit_code']==0
 reuse=json.loads((P/'OPS_NN_MASKEDFILL_SOURCE_REVIEW.json').read_text());assert reuse['host_API_reuse']['available'] is False;assert reuse['source_changes']==[]
 while True:
  assert not stopped()
  # User's02:00nightly gets worst-case600+150seconds before its start.
  nightly=nightly_check();state['nightly_window']=nightly
  if not nightly['allowed']:state.update(status='yield_0200_Shanghai_nightly_or_unobserved_dispatch',checked_at=now());put(A/'SERVER2_MASKEDSCATTER_GENERATION.json',state);time.sleep(60);continue
  okay,task=tasks();state.update(task_level=task,checked_at=now())
  if not okay:state['status']='CPU_wait_for_server2_task_release';put(A/'SERVER2_MASKEDSCATTER_GENERATION.json',state);time.sleep(60);continue
  meta_code='import pathlib,json,shutil,subprocess;print(json.dumps({"queue":[p.name for p in pathlib.Path("/run/lock/ascend-npu/device-0.queue").iterdir()],"workers":[x for x in subprocess.check_output(["ps","-eo","args"],text=True).splitlines() if x.split() and x.split()[0].endswith("Runner.Worker")],"disk":shutil.disk_usage("/").free}))'
  meta=json.loads(remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(meta_code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if meta['queue'] or meta['workers'] or meta['disk']<20000000000:state.update(status='yield_worker_queue_or_disk',meta=meta,checked_at=now());put(A/'SERVER2_MASKEDSCATTER_GENERATION.json',state);time.sleep(60);continue
  command='source '+ROOT+'/environment.sh; exec '+JOB+' '+GUARD+' status --json'
  r=remote(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
  if r.returncode:state.update(status='await_fresh_physical_idle',guard_status=r.stdout,guard_error=r.stderr,meta=meta,checked_at=now());put(A/'SERVER2_MASKEDSCATTER_GENERATION.json',state);time.sleep(30);continue
  g=json.loads(r.stdout)
  if g.get('state')!='idle' or g.get('reservation',{}).get('locked') is not False:state.update(status='await_fresh_physical_idle',guard_status=g,checked_at=now());put(A/'SERVER2_MASKEDSCATTER_GENERATION.json',state);time.sleep(30);continue
  # Recheck whole task release immediately before the guarded submission.
  okay,task=tasks()
  if not okay:state.update(status='CPU_wait_for_server2_task_release',task_level=task,checked_at=now());put(A/'SERVER2_MASKEDSCATTER_GENERATION.json',state);time.sleep(60);continue
  nightly=nightly_check();state['nightly_window']=nightly
  if not nightly['allowed']:state.update(status='yield_0200_Shanghai_nightly_or_unobserved_dispatch',checked_at=now());put(A/'SERVER2_MASKEDSCATTER_GENERATION.json',state);time.sleep(60);continue
  tag='server2-native-maskedscatter-device-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');receipt=N+'/'+tag+'-guard.json';gate={'checked_at':now(),'tasks':task,'meta':meta,'guard':g};put(A/(tag+'.GATE.json'),gate)
  cmd='source '+ROOT+'/environment.sh; exec '+JOB+' '+GUARD+' run --owner root/baseline-align/'+tag+' --wait 10 --timeout 600 --receipt '+receipt+' -- bash '+N+'/build-device-entry.sh'
  state.update(status='guarded_server2_reference_build',tag=tag,started_at=now());put(A/'SERVER2_MASKEDSCATTER_GENERATION.json',state)
  with (A/(tag+'.log')).open('w') as out:rc=remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
  got=remote('cat '+receipt,stdout=subprocess.PIPE);assert got.returncode==0;g=json.loads(got.stdout);(A/(tag+'-guard.json')).write_bytes(got.stdout)
  if rc==75 and g.get('phase')=='released_without_launch':state.update(status='deferred_no_child',checked_at=now());put(A/'SERVER2_MASKEDSCATTER_GENERATION.json',state);time.sleep(30);continue
  state.update(status='generated_pending_compile_scope_review' if rc==0 else 'reference_build_failed_requires_diagnosis',exit_code=rc,guard=g,checked_at=now(),log=tag+'.log')
  for f in [A/(tag+'.log'),A/(tag+'-guard.json')]:
   dst=N+'/'+f.name;code='import pathlib,sys;p=pathlib.Path('+repr(dst)+');data=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==data;p.write_bytes(data)';remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();state.setdefault('dual_archive',[]).append({'path':dst,'sha256':h})
  break
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:state['checked_at']=now();put(A/'SERVER2_MASKEDSCATTER_GENERATION.json',state)
