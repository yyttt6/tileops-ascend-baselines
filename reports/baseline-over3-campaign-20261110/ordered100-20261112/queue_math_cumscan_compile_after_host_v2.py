"""After terminal host success, inspect full commands, then two original commands per fresh guard."""
from pathlib import Path
import json,os,sys,time,subprocess,datetime,fcntl,hashlib,shlex
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-cumscan-v1';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();statefile=A/'MATH_CUMSCAN_COMPILE_AFTER_HOST_V2.json';s={'pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'status':'CPU_wait_MeanHost_terminal_success','queued_at':now(),'not_performance_or_coverage':True,'completed_stages':[]}
def put(p,v):
 t=p.with_name(p.name+'.compile.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def stopped():return (D/'STOP').exists() or (D.parent/'STOP').exists() or json.loads((D.parent/'STATE.json').read_text())['status']!='active'
def archive(p):
 raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();code='from pathlib import Path;import sys,hashlib;p=Path('+repr(N+'/'+p.name)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b);print(hashlib.sha256(b).hexdigest())';r=c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=raw,stdout=subprocess.PIPE,check=True);assert r.stdout.decode().strip()==h;return h
lock=(D/'math-cumscan-compile-after-host-v3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);put(statefile,s);switched=False
try:
 while True:
  assert not stopped();p=A/'SERVER1_MATH_CUMSCAN_GENERATION_V2.json'
  if p.exists():
   v=json.loads(p.read_text())
   if 'exit_code' in v or 'error' in v:assert v.get('exit_code')==0 and v['guard']['performance_valid'],v;break
   os.kill(v['pid'],0)
  time.sleep(10)
 assert not stopped();switched=True;s.update(status='CPU_full_source_and_whole_FP32_command_audit',checked_at=now());put(statefile,s);put(D/'CURRENT_PROCESS.json',{'status':'active_pipeline_build_not_measurement','pid':os.getpid(),'entry':Path(__file__).name,'phase':s['status'],'checked_at':now()})
 with (A/'math-cumscan-command-selection.log').open('w') as out:r=c.remote('/opt/tileops-ci/runtime/bin/python -I -S '+N+'/compile_math_cumscan_original.py prepare',stdout=out,stderr=subprocess.STDOUT)
 assert r.returncode==0,'No compiler called by CPU prepare; diagnose source/schema, no blindretry'
 raw=c.remote('cat '+N+'/CUMSCAN_COMPILE_SELECTION.json',stdout=subprocess.PIPE,check=True).stdout;(A/'MATH_CUMSCAN_COMPILE_SELECTION.initial.json').write_bytes(raw);count=len(json.loads(raw)['selected_whole_original_commands']);assert 3<=count<=48
 for stage in ['part'+str(i) for i in range((count+1)//2)]+['finalize']:
  while True:
   assert not stopped()
   with (D/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);okay,gate=c.admit()
    if okay:
     tag='server1-math-cumscan-'+stage+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');receipt=N+'/'+tag+'-guard.json';put(A/(tag+'.GATE.json'),gate);s.update(status='guarded_original_'+stage,tag=tag,checked_at=now());put(statefile,s)
     command='source '+ROOT+'/environment.sh; exec '+ROOT+'/job-python-v1/bin/python '+c.R+'/opportunistic_guard.py run --owner root/baseline-native-build/'+tag+' --wait 10 --timeout 600 --receipt '+receipt+' -- bash '+N+'/compile-entry.sh '+stage
     with (A/(tag+'.log')).open('w') as out:r=c.remote(command,stdout=out,stderr=subprocess.STDOUT)
     raw=c.remote('cat '+receipt,stdout=subprocess.PIPE,check=True).stdout;(A/(tag+'-guard.json')).write_bytes(raw);g=json.loads(raw)
    else:s.update(status='H1_resource_deferred_independent_of_H2',checked_at=now());put(statefile,s)
   if not okay:time.sleep(30);continue
   if r.returncode==75 and g.get('phase')=='released_without_launch':time.sleep(30);continue
   for p in [A/(tag+'.log'),A/(tag+'-guard.json')]:archive(p)
   s['last_guard']=g;s['last_exit_code']=r.returncode;put(statefile,s);assert r.returncode==0 and g['performance_valid'] is True and g['phase']=='finished' and g['workload_exit_code']==0,'Originalbuild failure preserved; no automaticretry'
   raw=c.remote('cat '+N+'/CUMSCAN_COMPILE_SELECTION.json',stdout=subprocess.PIPE,check=True).stdout;(A/('MATH_CUMSCAN_COMPILE_SELECTION.'+stage+'.json')).write_bytes(raw);s['completed_stages'].append(stage);s.update(checked_at=now());put(statefile,s);break
 s.update(status='original_all_float_commands_compiled_pending_package_fullAPIqualification',checked_at=now());put(statefile,s)
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e),checked_at=now());raise
finally:
 put(statefile,s)
 if switched:put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'outcome':s['status'],'checked_at':now()})
