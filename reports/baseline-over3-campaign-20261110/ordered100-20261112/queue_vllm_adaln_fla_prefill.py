"""Prefill2 via already qualified complete distinct vendor API; no oldcompile retry."""
from pathlib import Path
import json,sys,time,fcntl,subprocess,hashlib,shlex,os,datetime
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_vllm_adaln_fla_prefill import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'vllm-adaln-fla-prefill.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'queued_at':now(),'status':'CPU_source_fidelity','scope':[1120,1121],'accepted_alternate_APIs':0,'no_oldtiming_reuse':True};file=P/'VLLM_ADALN_FLA_PREFILL_EXPANSION.json';c.put(file,state)
try:
 assert not stop();old=json.loads((P/'VLLM_ADALN_FLA_DEFINED_CURRENT.json').read_text());assert old['accepted_alternate_APIs']==3
 proof=json.loads((P/'VLLM_ADALN_FLA_DEFINED_CPU_PROOF.json').read_text());assert proof['cases']==36 and len(proof['negative_contract_tests'])==7
 lineage=json.loads((P/'VLLM_ADALN_FLA_DEFINED_PROVENANCE.json').read_text());assert lineage['actual_git_blob']==lineage['expected_git_blob'] and lineage['kernel_body_not_edited']
 ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text())
 for i in state['scope']:assert 'MLIRCompilationError' in ledger[str(i)]['result']['error'] and 'hivm-plan-memory' in ledger[str(i)]['result']['error']
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  for rel in ['rollout_bootstrap.py','adaln_fla_defined_cached_fidelity.py','verify_adaln_fla_defined_cached.py','worker_vllm_adaln_fla_defined.py','vllm_adaln_fla_defined_support.py','workspace/isolated_wrappers/vllm_adaln_fla_defined_api.py']:
   f=P/rel;c.remote('cat > '+c.R+'/'+rel,input=f.read_bytes(),check=True);assert c.remote('sha256sum '+c.R+'/'+rel,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(f.read_bytes()).hexdigest()
 for i in state['scope']:
  with (P/f'adaln-fla-prefill-{i}.CPU.log').open('w') as out:r=c.remote('python3 '+c.R+'/verify_adaln_fla_defined_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT)
  assert r.returncode==0,'Source/payload preflight failed, no NPU submission'
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'distinct_complete_vllm_flaLayerNorm_prefill2_originalchecker15isolated','durable_log':'vllm-adaln-fla-prefill-queued.log'})
 for i in state['scope']:
  accepted=set(json.loads((P/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100'])
  if i in accepted:continue
  label='rollout-vllm-adaln-fla-prefill-'+str(i)+'-v1'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-vllm','adaln-fla-defined',[i],label)
   if e is not None:break
   state.update(status='resource_deferred',pending=i,checked_at=now());c.put(file,state);time.sleep(30)
  assert 'batch' in e and all(r['status']=='raw_checked_pending_guard' for r in e['batch']['results']),'NewAPI failurepreserved, no blindretry/expansion'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);v=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last=i,accepted_alternate_APIs=v['accepted_alternate_APIs'],checked_at=now());c.put(file,state)
 state['status']='prefill2_complete_full_originalchecker15raw'
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:state['checked_at']=now();c.put(file,state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':state['status']})
