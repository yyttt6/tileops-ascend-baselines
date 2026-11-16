"""Queue missing Norm coverage via a different complete vendor API, no old retries."""
import pathlib,sys,json,time,fcntl,subprocess,shlex,hashlib,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_vllm_norm_fla_defined import summarize
lock=(P/'vllm-norm-fla-defined.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
def predecessor_running():
 try:return b'queue_vllm_adaln_fla_defined.py' in pathlib.Path('/proc/1550030/cmdline').read_bytes()
 except FileNotFoundError:return False
state={'queued_at':stamp(),'status':'CPU_wait_for_predecessor','indices':[1195,1177,1180,1191,1192,1190,1194,1199],'batches':[[1195],[1177,1180,1191],[1192,1190,1194],[1199]],'pid':os.getpid(),'host':c.H,'representative':1195,'accepted_alternate_APIs':0,'candidate_only_no_old_timing_reuse':True,'existing_old_routes_remain_guard_blocked':True,'effective_coverage_addition_only_after_audit':True};c.put(P/'VLLM_NORM_FLA_DEFINED_EXPANSION.json',state)
try:
 while predecessor_running():assert not stopped();time.sleep(5)
 assert not stopped()
 prior=json.loads((P/'VLLM_ADALN_FLA_DEFINED_EXPANSION.json').read_text());assert prior['status']=='all_exact_scope_attempted_preserve_failures','Review shared/runtime fault before successor'
 proof=json.loads((P/'VLLM_NORM_FLA_DEFINED_CPU_PROOF.json').read_text());assert proof['cases']==32 and len(proof['negative_contract_tests'])==7
 lineage=json.loads((P/'VLLM_ADALN_FLA_DEFINED_PROVENANCE.json').read_text());assert lineage['actual_git_blob']==lineage['expected_git_blob'] and lineage['kernel_body_not_edited']
 ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text())
 for i in state['indices']:assert ledger[str(i)]['result']['status']=='unsupported' and 'Original LayerNorm sums masked-load lanes' in ledger[str(i)]['result']['error']
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);assert not stopped()
  f=P/'rollout_bootstrap.py';old=f.read_text();s=old
  if "'norm-fla-defined':" not in s:s=s.replace('script={',"script={'norm-fla-defined':'worker_vllm_norm_fla_defined.py',",1)
  f.with_name('rollout_bootstrap.before-norm-fla-defined.py').write_text(old);f.write_text(s)
  f=P/'continue_ordered.py';old=f.read_text();s=old;needle="'rollout-vllm-adaln-fla-defined-'";assert needle in s
  if "'rollout-vllm-norm-fla-defined-'" not in s:s=s.replace(needle,needle+",'rollout-vllm-norm-fla-defined-'",1)
  f.with_name('continue_ordered.before-norm-fla-defined.py').write_text(old);f.write_text(s)
  f=P/'effective_coverage.py';old=f.read_text();needle="'RESOLVED_ADALN_FLA_DEFINED_API.json'";assert needle in old
  if "'RESOLVED_NORM_FLA_DEFINED_API.json'" not in old:
   f.with_name('effective_coverage.before-norm-fla-defined.py').write_text(old);f.write_text(old.replace(needle,needle+",'RESOLVED_NORM_FLA_DEFINED_API.json'",1))
  records=[]
  for rel in ['rollout_bootstrap.py','norm_fla_defined_cached_fidelity.py','verify_norm_fla_defined_cached.py','worker_vllm_norm_fla_defined.py','vllm_norm_fla_defined_support.py','workspace/isolated_wrappers/vllm_norm_fla_defined_api.py']:
   f=P/rel;dst=c.R+'/'+rel;code='import pathlib,sys;p=pathlib.Path('+repr(dst)+');p.write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);actual=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==hashlib.sha256(f.read_bytes()).hexdigest();records.append({'path':rel,'sha256':actual})
  c.put(P/'VLLM_NORM_FLA_DEFINED_DEPLOYMENT.json',{'checked_at':stamp(),'files':records,'CPU_proof':'VLLM_NORM_FLA_DEFINED_CPU_PROOF.json','source_provenance':'VLLM_ADALN_FLA_DEFINED_PROVENANCE.json','protocol':'ascend950-isolated-ffts-v3','original_global_guard_unchanged':True,'no_old_control_repeated':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','updated_at':stamp(),'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'phase':'candidate_only_complete_defined_masked_LayerNorm_API_originalisolated15','durable_log':'vllm-norm-fla-defined-queued.log','snapshot_read_only':True});state.update(status='running',started_at=stamp());c.put(P/'VLLM_NORM_FLA_DEFINED_EXPANSION.json',state)
 pending=list(state['batches']);serial=0
 while pending:
  ids=pending.pop(0);serial+=1
  assert not stopped()
  for i in ids:
   with (P/f'norm-fla-defined-{i}-cached-source.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_norm_fla_defined_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'Cached source/payload qualification failed; no percase spin'
  label='rollout-vllm-norm-fla-defined-'+'-'.join(map(str,ids))+'-v1'
  while True:
   assert not stopped()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);event=c.run('r-vllm','norm-fla-defined',ids,label)
   if event is not None:break
   state.update(status='resource_deferred',pending_label=label,checked_at=stamp());c.put(P/'VLLM_NORM_FLA_DEFINED_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in event,'Invalid guard/process event preserved; requires diagnosis'
  for r in event['batch']['results']:
   if r.get('phase')=='factory' and not r.get('error','').startswith('UnsupportedCase('):raise RuntimeError('Unexpected factory/binding fault; stop shared group')
   if r.get('phase')=='load':raise RuntimeError('Shared load/source fault; stop affected group')
   if r['status']!='raw_checked_pending_guard':state.setdefault('failures',[]).append({'ordinal':r['ordinal'],'checkpoint':label+'.CHECKPOINT.json'})
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=ids,accepted_alternate_APIs=report['accepted_alternate_APIs'],checked_at=stamp());c.put(P/'VLLM_NORM_FLA_DEFINED_EXPANSION.json',state)
  attempted={r['ordinal'] for r in event['batch']['results']};assert attempted and attempted.issubset(set(ids))
  rest=[i for i in ids if i not in attempted]
  if rest:pending.insert(0,rest)
  if state['representative'] in ids:assert any(r['ordinal']==state['representative'] for r in report['rows']),'Representative failed; no expansion/retry'
 state['status']='all_exact_scope_attempted_preserve_failures'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=stamp();c.put(P/'VLLM_NORM_FLA_DEFINED_EXPANSION.json',state)
 if not predecessor_running():c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','checked_at':stamp(),'pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'outcome':state['status'],'next_action':'Review exact alternative coverage; dispatch qualified successor. Do not invent old/new ratios.','snapshot_read_only':True})
