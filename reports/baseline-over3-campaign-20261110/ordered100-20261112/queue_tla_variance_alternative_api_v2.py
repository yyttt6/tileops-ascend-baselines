from pathlib import Path
import sys,json,os,fcntl,time,datetime,subprocess,shlex,hashlib
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
from summarize_reducevar_tla_alternative_api_v2 import summarize
H='openlibing-DevEnv-661495';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-reducevar-original-v1';f=D/'TLA_VARIANCE_ALTERNATIVE_API_QUEUE_V2.json';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();s={'pid':os.getpid(),'entry':Path(__file__).name,'queued_at':now(),'status':'CPU_deploy_new_alternative_complete_API','scope':[2102, 2104, 2103, 2105, 2106, 2108, 2109, 2110, 2111, 2112]}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (D/'STOP').exists() or (D.parent/'STOP').exists() or json.loads((D.parent/'STATE.json').read_text())['status']!='active'
def rpc(cmd,**kw):return subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8',H,cmd],**kw)
def accepted():return set(json.loads((D/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100'])
with (D/'tla-variance-alternative-api-v2.lock').open('a') as own:
 fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not f.exists();put()
 try:
  assert not stop();hold=json.loads((A/'TLA_VARIANCE_SELECTED_RUNTIME_ISOLATION_V2.json').read_text());assert all(hashlib.sha256((D/name).read_bytes()).hexdigest()==value for name,value in hold['allowed_configuration_failure'].items());assert json.loads((A/'REDUCEVAR_ALTERNATIVE_PROBE_V1.json').read_text())['exit_code']==0
  files={name:(c.R,D/name) for name in ['rollout_bootstrap.py','worker_reducevar_tla_alternative_api_v2.py','reducevar_alternative_aot_retention.py','ops_math_reducevar_tla_alternative_api.py','reducevar_tla_exact_input_fidelity.py','REDUCEVAR_TLA_ALTERNATIVE_SCOPE.json','REDUCEVAR_TLA_ALTERNATIVE_FORMAL_PREREQUISITES.json']};files['runtime-env-formal-quiet-v4.sh']=(N,A/'reducevar-alternative-runtime-isolated-v2.sh')
  with (D/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);assert not stop()
   for name,(root,p) in files.items():
    raw=p.read_bytes();code='from pathlib import Path;import sys,hashlib;p=Path('+repr(root+'/'+name)+');raw=sys.stdin.buffer.read();p.write_bytes(raw);print(hashlib.sha256(raw).hexdigest())';r=rpc('python3 -I -S -c '+shlex.quote(code),input=raw,stdout=subprocess.PIPE,check=True);assert r.stdout.decode().strip()==hashlib.sha256(raw).hexdigest()
  for i in [2102, 2104, 2103, 2105, 2106, 2108, 2109, 2110, 2111, 2112]:
   if i in accepted():continue
   assert not list(D.glob('rollout-tla-variance-alternative-'+str(i)+'-h1-v2*.CHECKPOINT.json')),'Preserve prior formaloutcome, do not retry'
   while True:
    assert not stop()
    if i in accepted():break
    with (D/'controller.lock').open('a') as slot:
     fcntl.flock(slot,fcntl.LOCK_EX);assert not stop();c.H=H;okay,gate=c.admit()
     if okay:
      label='rollout-tla-variance-alternative-'+str(i)+'-h1-v2';s.update(status='guarded_formal_original_contract_alternative_API',ordinal=i,host=H,label=label);put();c.put(D/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'host':H,'ordinal':i,'phase':'originalTLAcontract_complete_ops_math_API15raw_not_originalTLAimplementation','checked_at':now()});event=c.run('r-template','tla-variance-alternative-api',[i],label)
     else:event=None
    if event is None:s.update(status='await_fresh_H1_admission',gates=gate);put();time.sleep(30);continue
    assert 'batch' in event and len(event['batch']['results'])==1 and event['batch']['results'][0]['status']=='raw_checked_pending_guard','Formal failure hold, no samecondition retry or sibling emptyloop'
    out=summarize();subprocess.run([sys.executable,str(D/'effective_coverage.py')],check=True);s.update(status='qualified_case_complete_next',last_ordinal=i,qualified_complete_APIs=out['accepted_complete_APIs']);put();break
  s['status']='eligible10_complete_API_cases_qualified_pending_pair_and_finalratio'
 except BaseException as e:s.update(status='stopped_requires_diagnosis_no_same_condition_retry',error=repr(e));raise
 finally:put();c.put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status']})
