"""Queue missing AdaLN coverage via a different complete vendor API, no old retries."""
import pathlib,sys,json,time,fcntl,subprocess,shlex,hashlib,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_vllm_adaln_fla_defined import summarize
lock=(P/'vllm-adaln-fla-defined.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
def predecessor_running():
 try:return b'queue_sgl_midpoint_api.py' in pathlib.Path('/proc/1526311/cmdline').read_bytes()
 except FileNotFoundError:return False
state={'queued_at':stamp(),'status':'CPU_wait_for_predecessor','indices':[1118,1119,1117],'representative':1118,'accepted_alternate_APIs':0,'candidate_only_no_old_timing_reuse':True,'existing_old_routes_remain_guard_blocked':True,'effective_coverage_addition_only_after_audit':True};c.put(P/'VLLM_ADALN_FLA_DEFINED_EXPANSION.json',state)
try:
 while predecessor_running():assert not stopped();time.sleep(5)
 assert not stopped()
 prior=json.loads((P/'SGL_MIDPOINT_API_EXPANSION.json').read_text());assert prior['status']=='all_exact_scope_attempted_preserve_failures','Review shared/runtime fault before successor'
 proof=json.loads((P/'VLLM_ADALN_FLA_DEFINED_CPU_PROOF.json').read_text());assert proof['cases']==36 and len(proof['negative_contract_tests'])==7
 lineage=json.loads((P/'VLLM_ADALN_FLA_DEFINED_PROVENANCE.json').read_text());assert lineage['actual_git_blob']==lineage['expected_git_blob'] and lineage['kernel_body_not_edited']
 ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text())
 for i in state['indices']:assert ledger[str(i)]['result']['status']=='unsupported' and 'Original LayerNorm sums masked-load lanes' in ledger[str(i)]['result']['error']
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);assert not stopped()
  f=P/'rollout_bootstrap.py';old=f.read_text();s=old
  if "'adaln-fla-defined':" not in s:s=s.replace('script={',"script={'adaln-fla-defined':'worker_vllm_adaln_fla_defined.py',",1)
  f.with_name('rollout_bootstrap.before-adaln-fla-defined.py').write_text(old);f.write_text(s)
  f=P/'continue_ordered.py';old=f.read_text();s=old;needle="'rollout-sgl-midpoint-api-candidate-'";assert needle in s
  if "'rollout-vllm-adaln-fla-defined-'" not in s:s=s.replace(needle,needle+",'rollout-vllm-adaln-fla-defined-'",1)
  f.with_name('continue_ordered.before-adaln-fla-defined.py').write_text(old);f.write_text(s)
  records=[]
  for rel in ['rollout_bootstrap.py','adaln_fla_defined_cached_fidelity.py','verify_adaln_fla_defined_cached.py','worker_vllm_adaln_fla_defined.py','vllm_adaln_fla_defined_support.py','workspace/isolated_wrappers/vllm_adaln_fla_defined_api.py']:
   f=P/rel;dst=c.R+'/'+rel;code='import pathlib,sys;p=pathlib.Path('+repr(dst)+');p.write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);actual=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==hashlib.sha256(f.read_bytes()).hexdigest();records.append({'path':rel,'sha256':actual})
  c.put(P/'VLLM_ADALN_FLA_DEFINED_DEPLOYMENT.json',{'checked_at':stamp(),'files':records,'CPU_proof':'VLLM_ADALN_FLA_DEFINED_CPU_PROOF.json','source_provenance':'VLLM_ADALN_FLA_DEFINED_PROVENANCE.json','protocol':'ascend950-isolated-ffts-v3','original_global_guard_unchanged':True,'no_old_control_repeated':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','updated_at':stamp(),'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'phase':'candidate_only_complete_defined_masked_LayerNorm_API_originalisolated15','durable_log':'vllm-adaln-fla-defined-queued.log','snapshot_read_only':True});state.update(status='running',started_at=stamp());c.put(P/'VLLM_ADALN_FLA_DEFINED_EXPANSION.json',state)
 for i in state['indices']:
  assert not stopped()
  with (P/f'adaln-fla-defined-{i}-cached-source.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_adaln_fla_defined_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'Cached source/payload qualification failed; no percase spin'
  label=f'rollout-vllm-adaln-fla-defined-{i}-v1'
  while True:
   assert not stopped()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);event=c.run('r-vllm','adaln-fla-defined',[i],label)
   if event is not None:break
   state.update(status='resource_deferred',pending_label=label,checked_at=stamp());c.put(P/'VLLM_ADALN_FLA_DEFINED_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in event,'Invalid guard/process event preserved; requires diagnosis'
  r=event['batch']['results'][0]
  if r.get('phase')=='load':raise RuntimeError('Shared load/source fault; stop affected group')
  if r['status']!='raw_checked_pending_guard':state.setdefault('failures',[]).append({'ordinal':i,'checkpoint':label+'.CHECKPOINT.json'})
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_ordinal=i,accepted_alternate_APIs=report['accepted_alternate_APIs'],checked_at=stamp());c.put(P/'VLLM_ADALN_FLA_DEFINED_EXPANSION.json',state)
  if i==state['representative']:assert report['accepted_alternate_APIs']==1,'Representative failed; no expansion/retry'
 state['status']='all_exact_scope_attempted_preserve_failures'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=stamp();c.put(P/'VLLM_ADALN_FLA_DEFINED_EXPANSION.json',state)
 if not predecessor_running():c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','checked_at':stamp(),'pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'outcome':state['status'],'next_action':'Review exact alternative coverage; dispatch qualified successor. Do not invent old/new ratios.','snapshot_read_only':True})
