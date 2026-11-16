from pathlib import Path
import sys,json,os,fcntl,time,datetime,subprocess
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
from summarize_tla_adaln7_math_alternative import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=D/'TLA_ADALN7_MATH_ALTERNATIVE_QUEUE_V2.json';s={'pid':os.getpid(),'entry':Path(__file__).name,'queued_at':now(),'status':'CPU_prerequisites','scope':[1620,1617,1618,1619,1621,1622,1623]}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (D/'STOP').exists() or (D.parent/'STOP').exists() or json.loads((D.parent/'STATE.json').read_text())['status']!='active'
def accepted():return set(json.loads((D/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100'])
with (D/'tla-adaln7-math-alternative-v2.lock').open('a') as own:
 fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not f.exists();put()
 try:
  proof=json.loads((A/'TLA_ADALN7_MATH_ALTERNATIVE_CPU_PROOF.json').read_text());assert len(proof['rows'])==7 and all(x['CPU_surrogate_matches_original_reference'] for x in proof['rows']);assert not stop()
  for i in s['scope']:
   if i in accepted():continue
   label='rollout-tla-adaln7-math-alternative-'+str(i)+'-h1-v2';assert not (D/(label+'.CHECKPOINT.json')).exists()
   while True:
    assert not stop()
    with (D/'controller.lock').open('a') as slot:
     fcntl.flock(slot,fcntl.LOCK_EX);assert not stop();s.update(status='fresh_H1_admission_or_guarded_formal_original_contract',ordinal=i,host=c.H,label=label);put();c.put(D/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'ordinal':i,'phase':'original_TLA_contract_complete_Math_APIs_isolated15','checked_at':now()});event=c.run('r-template','tla-adaln7-math-alternative',[i],label)
    if event is None:s.update(status='await_fresh_H1_admission');put();time.sleep(30);continue
    assert 'batch' in event and len(event['batch']['results'])==1 and event['batch']['results'][0]['status']=='raw_checked_pending_guard','Formal failure preserved; no samecondition retry or sibling emptyloop'
    out=summarize();subprocess.run([sys.executable,str(D/'effective_coverage.py')],check=True);s.update(status='qualified_case_complete_next',last_ordinal=i,qualified_complete_APIs=out['accepted_complete_APIs']);put();break
  s['status']='eligible7_complete_API_cases_qualified_pending_pair_and_finalratio'
 except BaseException as e:s.update(status='stopped_requires_diagnosis_no_same_condition_retry',error=repr(e));raise
 finally:put();c.put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status']})
