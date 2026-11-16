from pathlib import Path
import sys,json,os,fcntl,time,datetime,subprocess,shlex,hashlib
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
from summarize_math_bmm_transformer_alternative import summarize
H='openlibing-DevEnv-661495';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-sigmoid-original-v1';f=D/'MATH_BMM_TRANSFORMER_ALTERNATIVE_QUEUE_V2.json';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();s={'pid':os.getpid(),'entry':Path(__file__).name,'queued_at':now(),'status':'CPU_deploy_new_alternative_complete_API','scope':[242,243,246,247]}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (D/'STOP').exists() or (D.parent/'STOP').exists() or json.loads((D.parent/'STATE.json').read_text())['status']!='active'
def rpc(cmd,**kw):return subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8',H,cmd],**kw)
def accepted():return set(json.loads((D/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100'])
with (D/'math-bmm-transformer-alternative-v2.lock').open('a') as own:
 fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not f.exists();put()
 try:
  assert json.loads((A/'MATH_BMM_TRANSFORMER_ALTERNATIVE_CPU_PREREQUISITES.json').read_text())['all9_target_original_inputs_verified'];assert json.loads((A/'MATH_BMM_REFERENCE_HOLD_AND_CONTINUATION.json').read_text())['continue_ordinals']==[242,243,246,247];assert not stop()
  files={name:(c.R,D/name) for name in ['rollout_bootstrap.py','worker_math_bmm_transformer_alternative.py','math_bmm_transformer_alternative_api.py','math_bmm_transformer_alternative_fidelity.py']}
  with (D/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);assert not stop()
   for name,(root,p) in files.items():
    raw=p.read_bytes();code='from pathlib import Path;import sys,hashlib;p=Path('+repr(root+'/'+name)+');raw=sys.stdin.buffer.read();p.write_bytes(raw);print(hashlib.sha256(raw).hexdigest())';r=rpc('python3 -I -S -c '+shlex.quote(code),input=raw,stdout=subprocess.PIPE,check=True);assert r.stdout.decode().strip()==hashlib.sha256(raw).hexdigest()
  for i in [242,243,246,247]:
   if i in accepted():continue
   assert not list(D.glob('rollout-math-bmm-transformer-alternative-'+str(i)+'-h1-v1*.CHECKPOINT.json')),'Preserve prior formaloutcome, do not retry'
   while True:
    assert not stop()
    if i in accepted():break
    with (D/'controller.lock').open('a') as slot:
     fcntl.flock(slot,fcntl.LOCK_EX);assert not stop();c.H=H;okay,gate=c.admit()
     if okay:
      label='rollout-math-bmm-transformer-alternative-'+str(i)+'-h1-v1';s.update(status='guarded_formal_original_contract_alternative_API',ordinal=i,host=H,label=label);put();c.put(D/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'host':H,'ordinal':i,'phase':'originalMathcontract_completeOriginalTransformerAPI15raw_not_originalMathimplementation','checked_at':now()});event=c.run('r-template','math-bmm-transformer-alternative',[i],label)
     else:event=None
    if event is None:s.update(status='await_fresh_H1_admission',gates=gate);put();time.sleep(30);continue
    assert 'batch' in event and len(event['batch']['results'])==1 and event['batch']['results'][0]['status']=='raw_checked_pending_guard','Formal failure hold, no samecondition retry or sibling emptyloop'
    subprocess.run([sys.executable,str(D/'audit_math_bmm_transformer_alternative_raw.py')],check=True);out=summarize();subprocess.run([sys.executable,str(D/'effective_coverage.py')],check=True);s.update(status='qualified_case_complete_next',last_ordinal=i,qualified_complete_APIs=out['accepted_complete_APIs']);put();break
  s['status']='eligible4_remaining_complete_API_cases_qualified_pending_pair_and_finalratio'
 except BaseException as e:s.update(status='stopped_requires_diagnosis_no_same_condition_retry',error=repr(e));raise
 finally:put();c.put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status']})
