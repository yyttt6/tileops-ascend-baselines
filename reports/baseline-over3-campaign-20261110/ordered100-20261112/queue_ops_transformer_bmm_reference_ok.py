"""Remaining exact wide-axis cases; host-independent admission, stop on any real failure."""
from pathlib import Path
import sys,json,datetime,os,subprocess,shlex,fcntl,time,hashlib
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
from summarize_ops_transformer_bmm_api import summarize
from nightly_window_gate import check as nightly_check
from server2_whole_task_submission_gate import tasks as h2_tasks
H1='openlibing-DevEnv-661495';H2='openlibing-DevEnv-055409';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-arithmetic-v1';scope=json.loads((D/'TRANSFORMER_BMM_API_REFERENCE_OK_SCOPE.json').read_text())['ordinals'];now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=D/'OPS_TRANSFORMER_BMM_API_QUEUE.json';state={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_deploy_two_private_hosts','scope':scope,'queued_at':now(),'no_valid_result_repeats':True};ready=set()
def put():state['checked_at']=now();c.put(f,state)
def stop():return (D/'STOP').exists() or (D.parent/'STOP').exists() or json.loads((D.parent/'STATE.json').read_text())['status']!='active'
def completed():return set(json.loads((D/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100'])
def rpc(host,cmd,**kw):return subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8',host,cmd],**kw)
def deploy(host):
 if host in ready:return
 env=A/'transformer-bmm-api-runtime-env-formal-quiet-v3.sh';files={'runtime-env-formal-quiet-v3.sh':(N,env)}
 for name in ['rollout_bootstrap.py','ops_transformer_bmm_api_graph_api.py','ops_transformer_bmm_api_support.py','ops_transformer_bmm_api_cached_fidelity.py','worker_ops_transformer_bmm_api.py','TRANSFORMER_BMM_API_FORMAL_PREREQUISITES.json','TRANSFORMER_BMM_API_SCOPE.json','transformer_bmm_api_jit_retention.py','TRANSFORMER_BMM_API_QUALIFIED_JIT_CAPSULES.json']:files[name]=(c.R,D/name)
 rec={}
 for name,(root,p) in files.items():
  raw=p.read_bytes();code='from pathlib import Path;import sys,hashlib;p=Path('+repr(root+'/'+name)+');raw=sys.stdin.buffer.read();p.write_bytes(raw);print(hashlib.sha256(raw).hexdigest())';r=rpc(host,'python3 -I -S -c '+shlex.quote(code),input=raw,stdout=subprocess.PIPE,check=True);h=r.stdout.decode().strip();assert h==hashlib.sha256(raw).hexdigest();rec[name]=h
 c.put(D/('TRANSFORMER_BMM_API_DEPLOYMENT_'+('H1' if host==H1 else 'H2')+'.json'),{'checked_at':now(),'host':host,'files':rec,'timed_functions_AST_identical_first806':True,'LOG_only0_to3_no_compiler_or_kernelchanges':True});ready.add(host)
def prepare(host,i):
 code='from pathlib import Path;print((Path('+repr(c.R)+')/"prepared-rollout"/'+repr(str(i))+'/"prepare.json").is_file())';r=rpc(host,'python3 -I -S -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True)
 if r.stdout.strip()=='True':return
 with (D/('transformer-softmax-'+str(i)+'-'+('h1' if host==H1 else 'h2')+'.CPU.log')).open('w') as out:r=rpc(host,'bash '+c.R+'/launch.sh r-template prepare transformer-softmax-cpu-'+str(i)+' '+str(i),stdout=out,stderr=subprocess.STDOUT,timeout=600)
 assert r.returncode==0,'CPU original source/input preparation failed; no device retry'
lock=(D/'transformer-bmm-reference-ok.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);put()
try:
 assert not stop()
 with (D/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);p=D/'rollout_bootstrap.py';text=p.read_text()
  if "'ops-transformer-bmm-api':" not in text:
   old=D/'rollout_bootstrap.before-transformer-bmm-api-v1.py';assert not old.exists();old.write_text(text);p.write_text(text.replace('script={',"script={'ops-transformer-bmm-api':'worker_ops_transformer_bmm_api.py',",1))
 deploy(H1);deploy(H2)
 for i in scope:
  if i in completed():continue
  if i==810:continue  # Explicit preserved artifact-retention hold; no same-condition retry
  for cp in D.glob(f'rollout-ops-transformer-bmm-api-sdk-{i}*.CHECKPOINT.json'):
   old=json.loads(cp.read_text());assert 'batch' in old and all(r['status']=='raw_checked_pending_guard' for r in old['batch']['results']),'Prior failure hold; no crosshost blind retry'
  prepare(H1,i)
  while True:
   assert not stop()
   if i in completed():break
   chosen=None;gates={}
   with (D/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX)
    if i in completed():break
    c.H=H1;ok,gate=c.admit();gates['H1']=gate
    if ok:chosen=H1
    else:
     readiness=json.loads((A/'H2_TRANSFORMER_CPU_READINESS.json').read_text());gates['H2_CPU_readiness']=readiness;nightly=nightly_check() if readiness['ready'] else {'allowed':False,'reason':readiness['reason']};gates['H2_nightly']=nightly
     if nightly['allowed']:
      released,task=h2_tasks();gates['H2_task']=task
      if released:
       c.H=H2;ok,gate=c.admit();gates['H2']=gate
       if ok:chosen=H2
    if chosen is not None:
     c.H=chosen;prepare(chosen,i);assert not stop()
     proof=json.loads((A/'BMM_REMAINING_CPU_REFERENCE_PREFLIGHT.json').read_text());cert=next(x for x in proof['rows'] if x['ordinal']==i);assert cert['CPU_reference_preflight_ok']
     code='from pathlib import Path;import json;print(json.loads((Path('+repr(c.R)+')/"prepared-rollout"/'+repr(str(i))+'/"prepare.json").read_text())["payload_sha256"])';got=rpc(chosen,'python3 -I -S -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True);assert got.stdout.strip()==cert['payload_sha256'],'Preflightinputidentitymismatch; no nativeattempt'
     if chosen==H2:
      released,task=h2_tasks();gates['H2_fresh_task']=task;gates['H2_fresh_nightly']=nightly_check()
      if not released or not gates['H2_fresh_nightly']['allowed']:chosen=None
    if chosen is not None:
     hostlabel='h1' if chosen==H1 else 'h2';label=f'rollout-ops-transformer-bmm-api-sdk-{i}-{hostlabel}-v1';c.put(D/(label+'.TASK_GATE.json'),{'checked_at':now(),'gates':gates,'chosen_host':chosen});state.update(status='guarded_original_formal_case',current_ordinal=i,host=chosen,label=label);put();c.put(D/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'host':chosen,'ordinal':i,'phase':'original_Bmm_API_isolated15_fullchecker_ELF_capture','checked_at':now()});event=c.run('r-template','ops-transformer-bmm-api',[i],label)
    else:event=None
   if event is None:
    state.update(status='resource_deferred_no_launch',current_ordinal=i,gates=gates);put();c.put(D/'CURRENT_PROCESS.json',{'status':'waiting_fresh_per_host_admission','pid':os.getpid(),'entry':Path(__file__).name,'ordinal':i,'checked_at':now(),'not_effective_work':True});time.sleep(30);continue
   assert 'batch' in event,'Process/guard failure held; no blind continuation'
   rows=event['batch']['results'];assert len(rows)==1 and rows[0]['status']=='raw_checked_pending_guard','Original checker or parser failure held; do not loop sibling cases'
   subprocess.run([sys.executable,str(D/'audit_rollout.py')],check=True);r=summarize();raw=(D/'TRANSFORMER_BMM_API_QUALIFIED_JIT_CAPSULES.json').read_bytes();code='from pathlib import Path;import sys;Path('+repr(c.R+'/TRANSFORMER_BMM_API_QUALIFIED_JIT_CAPSULES.json')+').write_bytes(sys.stdin.buffer.read())';rpc(chosen,'python3 -I -S -c '+shlex.quote(code),input=raw,check=True);subprocess.run([sys.executable,str(D/'effective_coverage.py')],check=True);state.update(status='qualified_case_complete_next',last_ordinal=i,qualified_complete_APIs=r['accepted_complete_APIs']);put();break
 state['status']='all_eligible_wide_cases_qualified_original_checker15raw_ELF_guard_pending_API_review_and_denominators'
except BaseException as e:state.update(status='stopped_requires_diagnosis_no_same_condition_retry',error=repr(e));raise
finally:
 put();c.put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'outcome':state['status'],'checked_at':now()})
