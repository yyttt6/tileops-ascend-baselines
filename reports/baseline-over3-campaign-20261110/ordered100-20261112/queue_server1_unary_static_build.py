"""OriginalSDK staticresource packaging onindependentH1; H2nightly never blocks H1."""
import pathlib,json,datetime,fcntl,sys,subprocess,shlex,hashlib,os,time
P=pathlib.Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';sys.path.insert(0,str(P));import rollout_controller as c
ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';N=ROOT+'/native/ops-nn-unary-v1';JOB=ROOT+'/job-python-v1/bin/python'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'server1-unary-static-build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'host':c.H,'queued_at':now(),'status':'CPU_static_packaging_preflight','no_baseline_coverage_from_build':True,'max_guard_seconds':600,'server2_nightly_independent':True};c.put(A/'SERVER1_UNARY_STATIC_BUILD.json',state)
try:
 assert not stopped();assert json.loads((A/'UNARY_STATIC_KERNEL_INPUT_VERIFICATION.json').read_text())['original_kernel_objects_verified']==7;assert json.loads((A/'UNARY_DYNAMIC_DISPATCH_DIAGNOSIS.json').read_text())['private_original_device_kernels_verified'] is False
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'originalSDK_staticresource_packaging_no_newkernels_no_baselinecount','durable_log':'server1-unary-static-build-queued.log','snapshot_read_only':True})
 while True:
  assert not stopped()
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);okay,g=c.admit()
   if okay:
    tag='server1-unary-static-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');receipt=N+'/'+tag+'-guard.json';cmd='source '+ROOT+'/environment.sh; exec '+JOB+' '+c.R+'/opportunistic_guard.py run --owner root/baseline-unary-static/'+tag+' --wait 10 --timeout 600 --receipt '+receipt+' -- bash '+N+'/build-static-entry.sh';c.put(A/(tag+'.GATE.json'),g);state.update(status='guarded_original_static_resource_build',tag=tag,started_at=now());c.put(A/'SERVER1_UNARY_STATIC_BUILD.json',state)
    with (A/(tag+'.log')).open('w') as out:rc=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
    got=c.remote('cat '+receipt,stdout=subprocess.PIPE);guard=json.loads(got.stdout) if got.returncode==0 else {};assert got.returncode==0;(A/(tag+'-guard.json')).write_bytes(got.stdout)
   else:state.update(status='resource_deferred',checked_at=now());c.put(A/'SERVER1_UNARY_STATIC_BUILD.json',state)
  if not okay:time.sleep(30);continue
  if rc==75 and guard.get('phase')=='released_without_launch':time.sleep(30);continue
  state.update(status='static_archive_built_pending_link_and_actual_selection' if rc==0 else 'static_packaging_failed_requires_diagnosis',exit_code=rc,guard=guard,log=tag+'.log')
  for f in [A/(tag+'.log'),A/(tag+'-guard.json')]:
   dst=N+'/'+f.name;code='import pathlib,sys;p=pathlib.Path('+repr(dst)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b)';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();state.setdefault('dual_archive',[]).append({'path':dst,'sha256':h})
  break
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:
 state['checked_at']=now();c.put(A/'SERVER1_UNARY_STATIC_BUILD.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'next_action':'OriginalstaticAPI archives andembedded7payload identity/link/runtime proof required beforefrozenAbs/Neg15raw. H2schedulednightly taskrelease mustnotblockH1.','snapshot_read_only':True})
