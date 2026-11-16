"""First H2 scoped MathL1 formal after fulloriginalAPI/flush qualification. No crosshost ratio."""
from pathlib import Path
import os,sys,json,time,fcntl,subprocess,datetime
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_mathl1_h2 as c
from summarize_ops_math_l1_h2 import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
lock=(P/'ops-math-l1-controller.lock').open('a')
state={'pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'status':'CPU_source_fidelity','scope':[378,380,381,383],'queued_at':now(),'accepted_original_APIs':0,'not_paired_or_published':True};c.put(P/'OPS_MATH_MATHL1_H2_EXPANSION.json',state)
try:
 assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
 for i in state['scope']:
  with (P/f'mathl1-host2-{i}.CPU.log').open('w') as out:r=c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+__import__('shlex').quote("import sys,runpy;sys.path.insert(0,"+repr(c.R)+");sys.argv=["+repr(c.R+'/verify_ops_math_l1_cached.py')+","+repr(str(i))+"];runpy.run_path(sys.argv[0],run_name='__main__')"),stdout=out,stderr=subprocess.STDOUT)
  assert r.returncode==0,'H2source/payload mismatch, no NPU submission'
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'phase':'H2scope_fulloriginalAbs+Sum_originalisolated15','durable_log':'math-l1-h2-formal-v4-queued.log','updated_at':now()})
 for ids in [[378],[380,381,383]]:
  accepted=set(json.loads((P/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100']);ids=[i for i in ids if i not in accepted]
  if not ids:continue
  label='rollout-ops-math-l1h2-'+'-'.join(map(str,ids))+'-v2'
  while True:
   assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
   fcntl.flock(lock,fcntl.LOCK_EX)
   accepted=set(json.loads((P/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100']);ids=[i for i in ids if i not in accepted]
   if not ids:
    fcntl.flock(lock,fcntl.LOCK_UN);e=None;break
   assert not (P/'MATHL1_REQUIRES_DIAGNOSIS.json').exists(),'Otherhost failed; diagnose before crosshost retry'
   label='rollout-ops-math-l1h2-'+'-'.join(map(str,ids))+'-v2'
   e=c.run('r-template','mathl1-h2',ids,label)
   if e is not None:break
   fcntl.flock(lock,fcntl.LOCK_UN)
   state.update(status='fresh_H2_task_queue_guard_deferred',checked_at=now());c.put(P/'OPS_MATH_MATHL1_H2_EXPANSION.json',state);time.sleep(30)
  if not ids:continue
  if 'batch' not in e or not all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']):
   held=[x['ordinal'] for x in e.get('batch',{}).get('results',[]) if x['status']!='raw_checked_pending_guard'];c.put(P/'MATHL1_REQUIRES_DIAGNOSIS.json',{'ordinals':held or ids,'event':label,'reason':'Unqualified H2 result requires diagnosis before any crosshost retry; not automatically called numericfailure'});raise RuntimeError('Failure preserved, no blind H1/H2 retry')
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);r=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=ids,accepted_original_APIs=r['accepted_original_APIs'],checked_at=now());c.put(P/'OPS_MATH_MATHL1_H2_EXPANSION.json',state);fcntl.flock(lock,fcntl.LOCK_UN)
 state['status']='H2_full_original_API_scope_complete_no_crosshost_ratio'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=now();c.put(P/'OPS_MATH_MATHL1_H2_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'checked_at':now(),'outcome':state['status']})
