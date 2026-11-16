"""Original four frozen Cumprod contracts; percase full isolated15; no blind retries."""
from pathlib import Path
import sys,os,json,datetime,subprocess,shlex,fcntl,hashlib,time
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D));import rollout_controller as c
A=D/'server2-alignment-20261113';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=D/'OPS_MATH_CUMPROD_EXPANSION.json';state={'status':'CPU_deploy','pid':os.getpid(),'entry':Path(__file__).name,'scope':[252,253,254,255],'host':c.H,'started_at':now(),'not_qualified_until_full_AICPU_raw_and_guard_audit':True}
def put():state['checked_at']=now();c.put(f,state)
def stop():return (D/'STOP').exists() or (D.parent/'STOP').exists() or json.loads((D.parent/'STATE.json').read_text())['status']!='active'
lock=(D/'ops-math-cumprod-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);put()
try:
 assert not stop()
 with (D/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);p=D/'rollout_bootstrap.py';raw=p.read_text()
  if "'ops-math-cumprod-sdk-v1':" not in raw:
   old=D/'rollout_bootstrap.before-cumprod-sdk-v1.py';assert not old.exists();old.write_text(raw);p.write_text(raw.replace('script={',"script={'ops-math-cumprod-sdk-v1':'worker_ops_math_cumprod.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','ops_math_cumprod_api.py','ops_math_cumprod_support.py','ops_math_cumprod_cached_fidelity.py','worker_ops_math_cumprod.py','CUMPROD_FORMAL_PREREQUISITES.json']:
   p=D/name;dest=c.R+'/'+name;code='from pathlib import Path;import sys,hashlib;p=Path('+repr(dest)+');b=sys.stdin.buffer.read();p.write_bytes(b);print(hashlib.sha256(b).hexdigest())';h=c.remote('python3 -I -S -c '+shlex.quote(code),input=p.read_bytes(),stdout=subprocess.PIPE,check=True).stdout.decode().strip();assert h==hashlib.sha256(p.read_bytes()).hexdigest();records[name]=h
  c.put(D/'OPS_MATH_CUMPROD_DEPLOYMENT.json',{'checked_at':now(),'files':records,'kernel_flags_precision_checker_unchanged':True})
 c.put(D/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'phase':'original_Cumprod4_isolated_full32_AICPU_complete_API_pending_audit','checked_at':now()})
 for i in [252,253,254,255]:
  assert not stop();accepted=set(json.loads((D/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100'])
  if i in accepted:continue
  label=f'rollout-ops-math-cumprod-sdk-{i}-v1';cp=D/(label+'.CHECKPOINT.json');assert not cp.exists(),'No automatic repeat existing attempt'
  while True:
   assert not stop()
   with (D/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);event=c.run('r-template','ops-math-cumprod-sdk-v1',[i],label)
   if event is not None:break
   state.update(status='resource_deferred_no_launch',current_ordinal=i);put();time.sleep(30)
  assert 'batch' in event,'Guard/process failure held; do not loop cases'
  results=event['batch']['results'];state.update(status='raw_pending_separate_full_AICPU_attribution',current_ordinal=i,checkpoint=cp.name);put()
  assert len(results)==1 and results[0]['status']=='raw_checked_pending_guard','Full checker or trace failure held; no retry and no nextcase'
  subprocess.run([sys.executable,str(D/'audit_rollout.py')],check=True)
 state['status']='all4_raw_guard_audited_pending_AICPU_dependency_attribution_and_coverage'
except BaseException as e:state.update(status='stopped_requires_diagnosis_no_same_condition_retry',error=repr(e));raise
finally:
 put();c.put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'outcome':state['status'],'checked_at':now()})
