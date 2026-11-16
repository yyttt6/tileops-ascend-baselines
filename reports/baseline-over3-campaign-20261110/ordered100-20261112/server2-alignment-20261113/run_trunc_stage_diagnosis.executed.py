from pathlib import Path
import sys,json,datetime,hashlib,subprocess,shlex,fcntl,os
D=Path('/root/dyq/workspace/reports/baseline-over3-campaign-20261110/ordered100-20261112');A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-tensor-trunc-host-tile-v1';STACK='/opt/baseline-over3-20261110/server2-aligned-stack';JOB=STACK+'/job-python-v1/bin/python';tag='server1-trunc-stage-diagnosis-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');s={'status':'CPU_preflight','pid':os.getpid(),'entry':Path(__file__).name,'scope':'one8192element diagnostic fixture','not_acceptance_or_coverage':True,'tag':tag,'started_at':now()};p=A/'TRUNC_SIGNED_ZERO_STAGE_DIAGNOSIS.json'
def put():s['checked_at']=now();c.put(p,s)
put()
try:
 assert json.loads((D.parent/'STATE.json').read_text())['status']=='active'
 source=A/'diagnose_trunc_signed_zero_stages.py';raw=source.read_bytes();compile(raw,str(source),'exec');dest=N+'/'+source.name;code='from pathlib import Path;import sys,hashlib;p=Path('+repr(dest)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b);print(hashlib.sha256(b).hexdigest())';got=c.remote('python3 -I -S -c '+shlex.quote(code),input=raw,stdout=subprocess.PIPE,check=True).stdout.decode().strip();assert got==hashlib.sha256(raw).hexdigest();s['probe_sha256']=got
 with (D/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);okay,gate=c.admit();s['gate']=gate
  if not okay:s['status']='deferred_no_launch';put();sys.exit(75)
  receipt=N+'/'+tag+'-guard.json';owner='root/baseline-native-probe/'+tag;result=tag+'.json';cmd='source '+STACK+'/native/ops-math-layernorm-original-v1/runtime-env-formal.sh; exec '+JOB+' '+c.R+'/opportunistic_guard.py run --owner '+owner+' --wait 10 --timeout 90 --receipt '+receipt+' -- '+JOB+' '+dest+' '+owner+' '+result
  s['status']='guarded_stage_diagnosis';put();c.put(D/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'phase':'diagnostic_stage_capture_no15raw','checked_at':now()})
  with (A/(tag+'.log')).open('w') as f:r=c.remote(cmd,stdout=f,stderr=subprocess.STDOUT)
  s['exit_code']=r.returncode;s['artifacts']=[]
  for name in [tag+'-guard.json',result]:
   data=c.remote('cat '+shlex.quote(N+'/'+name),stdout=subprocess.PIPE,check=True).stdout;(A/name).write_bytes(data);s['artifacts'].append({'path':name,'sha256':hashlib.sha256(data).hexdigest()})
  for name in [tag+'.log']:
   data=(A/name).read_bytes();dest=N+'/'+name;code='from pathlib import Path;import sys,hashlib;p=Path('+repr(dest)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b);print(hashlib.sha256(b).hexdigest())';got=c.remote('python3 -I -S -c '+shlex.quote(code),input=data,stdout=subprocess.PIPE,check=True).stdout.decode().strip();assert got==hashlib.sha256(data).hexdigest();s['artifacts'].append({'path':name,'sha256':got})
  s['status']='stage_capture_completed_not_acceptance' if r.returncode==0 else 'diagnostic_failed_no_retry';s['result']=result
except BaseException as e:s['error']=repr(e);raise
finally:
 put();c.put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'outcome':s['status'],'checked_at':now()})
print(json.dumps(s))
