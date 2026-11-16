"""Qualify traceable server2 artifacts in aligned private server1; no shared install."""
import pathlib,subprocess,json,sys,os,fcntl,datetime,hashlib,shlex,time
P=pathlib.Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';sys.path.insert(0,str(P));import rollout_controller as c
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';JOB=ROOT+'/job-python-v1/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
 t=p.with_name(p.name+'.load.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'server1-native-load.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'entry':pathlib.Path(__file__).name,'pid':os.getpid(),'host':c.H,'queued_at':now(),'status':'CPU_artifact_deployment','no_baseline_or_numeric_acceptance':True};put(A/'SERVER1_NATIVE_LOAD.json',state)
try:
 assert not stop();prior=json.loads((P/'TILELANG_RMS_EXPANSION.json').read_text());assert prior['status']=='all_exact_scope_attempted_preserve_failures' and prior['accepted_original_APIs']==18
 assert not pathlib.Path('/proc/1569670/cmdline').exists()
 m=json.loads((A/'ARTIFACT_MANIFEST.server2.json').read_text());assert m['source_changes']==[] and m['source_tracked_files_verified']==25610
 artifact='ARTIFACTS-server2-aligned-native-162714998720.json';stage=pathlib.Path(m['staging_root']).relative_to(N);archive=A/'ARTIFACT_PACKAGE.server2.tar.gz'
 if not archive.exists():
  with archive.open('wb') as out:subprocess.run(['ssh','openlibing-DevEnv-055409','tar -czf - -C '+N+' '+artifact+' load_native_package.py '+str(stage)],stdout=out,check=True)
 checksum=sha(archive);remote_archive=N+'/qualification-artifacts.tar.gz'
 code='import pathlib,sys;p=pathlib.Path('+repr(remote_archive)+');data=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==data;p.write_bytes(data)';c.remote('python3 -c '+shlex.quote(code),input=archive.read_bytes(),check=True)
 # CPU-only extraction of data files into one taskprivate namespace; no installer runs.
 code='''import pathlib,tarfile,hashlib,json
n=pathlib.Path(%r);a=n/'qualification-artifacts.tar.gz';assert hashlib.sha256(a.read_bytes()).hexdigest()==%r
with tarfile.open(a) as t:
 for e in t.getmembers():
  p=pathlib.PurePosixPath(e.name);assert not p.is_absolute() and '..' not in p.parts;dest=n/e.name
  if e.isfile() and dest.exists():assert dest.read_bytes()==t.extractfile(e).read()
 t.extractall(n,filter='data')
m=json.loads((n/%r).read_text());assert not m['source_changes'];root=pathlib.Path(m['staging_root'])
for rel,e in m['staged_files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==e['sha256']
print(json.dumps({'archive_sha256':%r,'staged_files_verified':len(m['staged_files']),'manifest_sha256':hashlib.sha256((n/%r).read_bytes()).hexdigest(),'no_shared_installation':True}))
'''%(N,checksum,artifact,checksum,artifact)
  
 r=c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True);deployment=json.loads(r.stdout);assert deployment['manifest_sha256']==sha(A/'ARTIFACT_MANIFEST.server2.json');put(A/'ARTIFACT_DEPLOYMENT.server1.json',deployment)
 put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'aligned_private_native_DSO_load_qualification','durable_log':'server2-alignment-20261113/server1-native-load-queued.log','snapshot_read_only':True})
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);okay,g=c.admit()
   if okay:
    tag='server1-native-load-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');owner='root/baseline-native-load/'+tag;receipt=N+'/'+tag+'-guard.json';result=tag+'.json';root=m['staging_root'];dirs=[root+'/op_api/lib',root+'/op_proto/es/lib/linux/aarch64',root+'/op_proto/lib/linux/aarch64',root+'/op_impl/ai_core/tbe/op_tiling'];env='env ASCEND_CUSTOM_OPP_PATH='+shlex.quote(root)+' LD_LIBRARY_PATH='+':'.join(dirs)+':"$LD_LIBRARY_PATH" '
    cmd='source '+ROOT+'/environment.sh; exec '+JOB+' '+c.R+'/opportunistic_guard.py run --owner '+owner+' --wait 10 --timeout 120 --receipt '+receipt+' -- '+env+JOB+' '+N+'/load_native_package.py '+owner+' '+result
    put(A/(tag+'.GATE.json'),g);state.update(status='guarded_native_load',tag=tag,started_at=now());put(A/'SERVER1_NATIVE_LOAD.json',state)
    with (A/(tag+'.log')).open('w') as out:rc=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
    r=c.remote('cat '+receipt,stdout=subprocess.PIPE,text=True);guard=json.loads(r.stdout) if r.returncode==0 else {};put(A/(tag+'-guard.json'),guard)
   else:state.update(status='resource_deferred',checked_at=now());put(A/'SERVER1_NATIVE_LOAD.json',state)
  if not okay:time.sleep(30);continue
  if rc==75 and guard.get('phase')=='released_without_launch':time.sleep(30);continue
  r=c.remote('cat '+N+'/'+result,stdout=subprocess.PIPE,text=True);data=json.loads(r.stdout) if r.returncode==0 else {};put(A/result,data);state.update(status='loaded_pending_numeric_qualification' if rc==0 and guard.get('performance_valid') else 'load_failed_requires_diagnosis',exit_code=rc,guard=guard,results=data.get('results',[]),result=result)
  for f in [A/(tag+'.log'),A/(tag+'-guard.json'),A/result]:
   dest=N+'/'+f.name;code='import pathlib,sys;p=pathlib.Path('+repr(dest)+');data=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==data;p.write_bytes(data)';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==sha(f);state.setdefault('dual_archive',[]).append({'path':dest,'sha256':h})
  break
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:
 state['checked_at']=now();put(A/'SERVER1_NATIVE_LOAD.json',state);put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
