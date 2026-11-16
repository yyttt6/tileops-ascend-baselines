"""Bounded private CMake diagnostic, after actual predecessor exit; never a benchmark."""
import pathlib,sys,json,time,fcntl,subprocess,shlex,hashlib,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'opsnn-private-configure2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
def predecessor():
 try:return b'queue_opsnn_private_configure.py' in pathlib.Path('/proc/1499838/cmdline').read_bytes()
 except FileNotFoundError:return False
state={'queued_at':now(),'status':'CPU_wait_for_predecessor','entry':pathlib.Path(__file__).name,'pid':os.getpid(),'purpose':'private source CMake configure only; no native acceptance or coverage','predecessor_pid':1499838,'timeout_seconds':120}
c.put(P/'OPS_NN_PRIVATE_CONFIGURE2.json',state)
try:
 while predecessor():
  assert not stopped();time.sleep(5)
 assert not stopped()
 prior=json.loads((P/'OPS_NN_PRIVATE_CONFIGURE.json').read_text())
 assert prior['status']=='configure_failed_requires_dependency_diagnosis' and prior['exit_code']==1
 assert json.loads((P/'OPS_NN_ASC_PACKAGE_DIAGNOSIS.json').read_text())['source_edits'] is False
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);assert not stopped()
  local=P/'opsnn-private-configure2.sh';destination=c.R+'/native-rebuild/ops-nn-v1/configure2.sh'
  code='import pathlib,sys;p=pathlib.Path('+repr(destination)+');assert not p.exists();p.write_bytes(sys.stdin.buffer.read())'
  c.remote('python3 -c '+shlex.quote(code),input=local.read_bytes(),check=True)
  actual=c.remote('sha256sum '+destination,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]
  assert actual==hashlib.sha256(local.read_bytes()).hexdigest()
  state.update(script_sha256=actual,started_at=now(),status='await_fresh_idle')
  c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'bounded_private_native_configure_no_performance','snapshot_read_only':True})
  while True:
   assert not stopped()
   tag='opsnn-private-configure2-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f')
   okay,gate=c.admit();c.put(P/(tag+'.GATE.json'),gate)
   if not okay:time.sleep(30);continue
   receipt=c.R+'/'+tag+'-guard.json'
   cmd=c.ENV+c.PY+c.R+'/opportunistic_guard.py run --owner root/baseline-ordered100/'+tag+' --wait 10 --timeout 120 --receipt '+receipt+' -- bash '+destination
   state.update(status='guarded_private_configure',tag=tag,checked_at=now());c.put(P/'OPS_NN_PRIVATE_CONFIGURE2.json',state)
   with (P/(tag+'.log')).open('w') as out:rc=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
   fetched=c.remote('cat '+receipt,stdout=subprocess.PIPE,text=True)
   guard=json.loads(fetched.stdout) if fetched.returncode==0 else {};c.put(P/(tag+'-guard.json'),guard)
   if rc==75 and guard.get('phase')=='released_without_launch':continue
   state.update(status='configure_finished' if rc==0 else 'configure_failed_requires_dependency_diagnosis',exit_code=rc,guard=guard,checked_at=now(),log=tag+'.log')
   # Preserve log and result on both hosts; cache is a diagnostic, no installed binary is claimed.
   for path in [P/(tag+'.log'),P/(tag+'-guard.json')]:
    dst=c.R+'/native-rebuild/ops-nn-v1/'+path.name
    code='import pathlib,sys;p=pathlib.Path('+repr(dst)+');assert not p.exists();p.write_bytes(sys.stdin.buffer.read())'
    c.remote('python3 -c '+shlex.quote(code),input=path.read_bytes(),check=True)
    sha=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert sha==hashlib.sha256(path.read_bytes()).hexdigest()
    state.setdefault('dual_archive',[]).append({'local':path.name,'remote':dst,'sha256':sha})
   cache=c.remote('cat '+c.R+'/native-rebuild/ops-nn-v1/build-config1/CMakeCache.txt',stdout=subprocess.PIPE)
   if cache.returncode==0:(P/'OPS_NN_CMakeCache.configure2.txt').write_bytes(cache.stdout)
   break
except BaseException as exc:
 state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=now();c.put(P/'OPS_NN_PRIVATE_CONFIGURE2.json',state)
 if not predecessor():c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','checked_at':now(),'pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'outcome':state['status'],'next_action':'Review CMake dependency diagnostic; no repeated failed configure or performance count','snapshot_read_only':True})
