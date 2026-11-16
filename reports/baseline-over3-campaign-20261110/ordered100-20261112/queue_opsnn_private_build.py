"""Native source support job serialized at the next available batch boundary."""
import pathlib,sys,fcntl,json,datetime,os,time,shlex,hashlib,subprocess
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'opsnn-private-build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'status':'CPU_wait_for_batch_boundary','queued_at':stamp(),'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'timeout_seconds':600,'no_coverage_or_performance_acceptance':True,'does_not_overwrite_active_benchmark_controller':True}
c.put(P/'OPS_NN_PRIVATE_BUILD.json',state)
try:
 assert json.loads((P/'OPS_NN_FULL_CONFIGURE.json').read_text())['status']=='full_configure_passed'
 while True:
  assert not stopped()
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);assert not stopped()
   okay,gate=c.admit();tag='opsnn-private-build3-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');c.put(P/(tag+'.GATE.json'),gate)
   if okay:
    src=P/'opsnn-private-build1.sh';dest=c.R+'/native-rebuild/ops-nn-v1/build1.sh'
    query='import pathlib,sys;p=pathlib.Path('+repr(dest)+');data=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==data;p.write_bytes(data)'
    c.remote('python3 -c '+shlex.quote(query),input=src.read_bytes(),check=True)
    digest=c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert digest==hashlib.sha256(src.read_bytes()).hexdigest()
    receipt=c.R+'/'+tag+'-guard.json';cmd=c.ENV+c.PY+c.R+'/opportunistic_guard.py run --owner root/baseline-ordered100/'+tag+' --wait 10 --timeout 600 --receipt '+receipt+' -- bash '+dest
    state.update(status='guarded_original_private_build',tag=tag,started_at=stamp(),script_sha256=digest);c.put(P/'OPS_NN_PRIVATE_BUILD.json',state)
    with (P/(tag+'.log')).open('w') as out:rc=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
    q=c.remote('cat '+receipt,stdout=subprocess.PIPE,text=True);guard=json.loads(q.stdout) if q.returncode==0 else {};c.put(P/(tag+'-guard.json'),guard)
    if rc==75 and guard.get('phase')=='released_without_launch':state.update(status='waiting_fresh_idle_no_child',checked_at=stamp())
    else:
     state.update(status='private_build_passed' if rc==0 else 'private_build_failed_requires_diagnosis',exit_code=rc,guard=guard,checked_at=stamp(),log=tag+'.log')
     for f in [P/(tag+'.log'),P/(tag+'-guard.json')]:
      target=c.R+'/native-rebuild/ops-nn-v1/'+f.name;query='import pathlib,sys;p=pathlib.Path('+repr(target)+');assert not p.exists();p.write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(query),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+target,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();state.setdefault('dual_archive',[]).append({'local':f.name,'remote':target,'sha256':h})
     cache=c.remote('cat '+c.R+'/native-rebuild/ops-nn-v1/source/build-config3/CMakeCache.txt',stdout=subprocess.PIPE)
     if cache.returncode==0:(P/'OPS_NN_CMakeCache.build1.txt').write_bytes(cache.stdout)
     break
   else:state.update(status='await_fresh_idle',checked_at=stamp())
  c.put(P/'OPS_NN_PRIVATE_BUILD.json',state);time.sleep(30)
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:state['checked_at']=stamp();c.put(P/'OPS_NN_PRIVATE_BUILD.json',state)
