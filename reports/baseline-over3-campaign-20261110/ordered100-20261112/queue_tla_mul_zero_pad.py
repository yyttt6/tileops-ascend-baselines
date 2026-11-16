from pathlib import Path
import json,os,sys,datetime,fcntl,subprocess,hashlib,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_tla_mul_zero_pad import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();lock=(P/'tla-mul-zero-pad.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_control_source_identity','scope':[2002,2003],'host':c.H,'queued_at':now(),'coverage_increment':0,'original_control_reused_no_rerun':True};file=P/'TLA_MUL_ZERO_PAD_EXPANSION.json';c.put(file,state)
try:
 assert not (P/'STOP').exists() and not (P.parent/'STOP').exists();proof=json.loads((P/'TLA_MUL_ZERO_PAD_CONTROL_PREFLIGHT.json').read_text());assert proof['checked_files']==165 and not proof['mismatches'] and proof['protocol']=='experimental-baseline-full32-marker-session-v1'
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);p=P/'rollout_bootstrap.py';s=p.read_text()
  if "'tla-mul-zero-pad':" not in s:p.with_name('rollout_bootstrap.before-mul-zero-pad.py').write_text(s);p.write_text(s.replace('script={',"script={'tla-mul-zero-pad':'worker_tla_mul_zero_pad.py',",1))
  for rel in ['rollout_bootstrap.py','worker_tla_mul_zero_pad.py','tla_mul_zero_pad_support.py','tla_mul_zero_pad_cached_fidelity.py','verify_tla_mul_zero_pad_cached.py','workspace/isolated_wrappers/tla_zero_pad_candidate.py']:
   f=P/rel;c.remote('cat > '+c.R+'/'+rel,input=f.read_bytes(),check=True);assert c.remote('sha256sum '+c.R+'/'+rel,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(f.read_bytes()).hexdigest()
 for i in state['scope']:
  with (P/f'tla-mul-zero-pad-{i}.CPU.log').open('w') as out:r=c.remote('python3 '+c.R+'/verify_tla_mul_zero_pad_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT)
  assert r.returncode==0,'Source/payload preflight failed; no percase spin'
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'phase':'Mul_zero_pad_candidate2_same_experimentalprotocol15raw_completechecker','updated_at':now(),'durable_log':'tla-mul-zero-pad-v2-queued.log'})
 for i in state['scope']:
  label='rollout-tla-mul-zero-pad-'+str(i)+'-v1'
  while True:
   assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
   with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-tla-original','tla-mul-zero-pad',[i],label)
   if e is not None:break
   state.update(status='resource_deferred',pending=i,checked_at=now());c.put(file,state);time.sleep(30)
  assert 'batch' in e and all(r['status']=='raw_checked_pending_guard' for r in e['batch']['results']),'Candidate failure retained, no blindretry'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();state.update(status='running',paired_cases=report['paired_cases'],checked_at=now());c.put(file,state)
 state['status']='candidate2_fullchecker15raw_audited_comparisons_complete'
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:state['checked_at']=now();c.put(file,state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':state['status']})
