"""Qualify traceable server2 artifacts in aligned private server1; no shared install."""
import pathlib,subprocess,json,sys,os,fcntl,datetime,hashlib,shlex,time
P=pathlib.Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';sys.path.insert(0,str(P));import rollout_controller as c
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';JOB=ROOT+'/job-python-v1/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
 t=p.with_name(p.name+'.load.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def qualified_probe(rc,guard,data):
 rows=data.get('rows',[])
 return rc==0 and guard.get('performance_valid') is True and data.get('status')=='functional_probe_passed_not_formal_baseline' and len(rows)==3 and {r.get('dtype') for r in rows}=={'torch.float16','torch.bfloat16','torch.float32'} and all(r.get('status')=='full_outputs_and_inputs_verified' and r.get('outputs')==2 for r in rows)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'server1-native-probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'entry':pathlib.Path(__file__).name,'pid':os.getpid(),'host':c.H,'queued_at':now(),'status':'CPU_artifact_deployment','no_formal_baseline_acceptance':True,'functional_API_probe_only':True};put(A/'SERVER1_NATIVE_PROBE.json',state)
try:
 assert not stop();prior=json.loads((P/'TILELANG_RMS_EXPANSION.json').read_text());assert prior['status']=='all_exact_scope_attempted_preserve_failures' and prior['accepted_original_APIs']==18
 assert not pathlib.Path('/proc/1569670/cmdline').exists()
 priorload=json.loads((A/'SERVER1_NATIVE_LOAD.json').read_text());assert priorload['status']=='loaded_pending_numeric_qualification' and priorload['guard']['performance_valid'] and priorload['guard']['workload_exit_code']==0
 m=json.loads((A/'ARTIFACT_MANIFEST.server2.json').read_text());assert m['source_changes']==[]
 f=A/'probe_foreach_native.py';dst=N+'/'+f.name;code='import pathlib,sys;pathlib.Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);assert c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==sha(f)
 put(A/'NATIVE_PROBE_DEPLOYMENT.server1.json',{'checked_at':now(),'probe_sha256':sha(f),'artifact_manifest_sha256':sha(A/'ARTIFACT_MANIFEST.server2.json'),'no_shared_runtime_or_kernel_change':True})
 put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'complete_native_ForeachMulList_functional_probe_no_formalbaseline_count','durable_log':'server2-alignment-20261113/server1-native-probe-queued.log','snapshot_read_only':True})
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);okay,g=c.admit()
   if okay:
    tag='server1-native-functional-v2-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');owner='root/baseline-native-probe/'+tag;receipt=N+'/'+tag+'-guard.json';result=tag+'.json';root=m['staging_root'];dirs=[root+'/op_api/lib',root+'/op_proto/es/lib/linux/aarch64',root+'/op_proto/lib/linux/aarch64',root+'/op_impl/ai_core/tbe/op_tiling'];env='env ASCEND_CUSTOM_OPP_PATH='+shlex.quote(root)+' LD_LIBRARY_PATH='+':'.join(dirs)+':"$LD_LIBRARY_PATH" '
    cmd='source '+ROOT+'/environment.sh; exec '+JOB+' '+c.R+'/opportunistic_guard.py run --owner '+owner+' --wait 10 --timeout 600 --receipt '+receipt+' -- '+env+JOB+' '+N+'/probe_foreach_native.py '+owner+' '+result
    put(A/(tag+'.GATE.json'),g);state.update(status='guarded_native_probe',tag=tag,started_at=now());put(A/'SERVER1_NATIVE_PROBE.json',state)
    with (A/(tag+'.log')).open('w') as out:rc=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
    r=c.remote('cat '+receipt,stdout=subprocess.PIPE,text=True);guard=json.loads(r.stdout) if r.returncode==0 else {};put(A/(tag+'-guard.json'),guard)
   else:state.update(status='resource_deferred',checked_at=now());put(A/'SERVER1_NATIVE_PROBE.json',state)
  if not okay:time.sleep(30);continue
  if rc==75 and guard.get('phase')=='released_without_launch':time.sleep(30);continue
  r=c.remote('cat '+N+'/'+result,stdout=subprocess.PIPE,text=True);data=json.loads(r.stdout) if r.returncode==0 else {};put(A/result,data);state.update(status='functional_probe_passed_not_formal_baseline' if qualified_probe(rc,guard,data) else 'functional_probe_failed_requires_diagnosis',exit_code=rc,guard=guard,results=data.get('results',[]),result=result)
  for f in [A/(tag+'.log'),A/(tag+'-guard.json'),A/result]:
   dest=N+'/'+f.name;code='import pathlib,sys;p=pathlib.Path('+repr(dest)+');data=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==data;p.write_bytes(data)';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==sha(f);state.setdefault('dual_archive',[]).append({'path':dest,'sha256':h})
  break
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:
 state['checked_at']=now();put(A/'SERVER1_NATIVE_PROBE.json',state);put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
