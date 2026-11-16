"""CPU-queued successor; no source deployment or device work before predecessor exits."""
import pathlib,sys,json,time,fcntl,subprocess,shlex,hashlib,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_vllm_bmm_linear import summarize
own=(P/'vllm-bmm-linear.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def predecessor_running():
 try:return b'queue_vllm_gemm_linear.py' in pathlib.Path('/proc/1439636/cmdline').read_bytes()
 except FileNotFoundError:return False
state={'queued_at':stamp(),'indices':[1150,1151,1144,1145,1142,1143,1138,1139,1140,1141,1137,1136,1149,1148,1147,1146],'representative':1150,'complete_pairs':0,'status':'CPU_wait_for_predecessor','no_predictive_memory_gate':True}
c.put(P/'VLLM_BMM_LINEAR_EXPANSION.json',state)
try:
 while predecessor_running():
  assert not stopped();time.sleep(5)
 assert not stopped()
 prior=json.loads((P/'VLLM_GEMM_LINEAR_EXPANSION.json').read_text());assert prior['status'] not in ('running','preparing','resource_deferred','stopped_requires_diagnosis'),'predecessor shared fault/final state requires review before same-dependency dispatch'
 assert json.loads((P/'VLLM_BMM_LINEAR_CPU_PROOF.json').read_text())['cases']==12
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);assert not stopped()
  b=P/'rollout_bootstrap.py';old=b.read_text();new=old
  if "'bmm-linear-original':" not in old:new=old.replace("script={","script={'bmm-linear-original':'worker_bmm_linear_original.py','bmm-linear-candidate':'worker_vllm_bmm_linear.py',",1)
  b.with_name('rollout_bootstrap.before-bmm-linear.py').write_text(old);b.write_text(new)
  b=P/'continue_ordered.py';old=b.read_text();needle="'rollout-vllm-gemm-linear-candidate-'";assert needle in old
  if "'rollout-vllm-bmm-linear-candidate-'" not in old:b.with_name('continue_ordered.before-bmm-linear.py').write_text(old);b.write_text(old.replace(needle,needle+",'rollout-vllm-bmm-linear-candidate-'",1))
  records=[]
  for rel in ['rollout_bootstrap.py','bmm_linear_cached_fidelity.py','verify_bmm_linear_cached.py','worker_bmm_linear_original.py','worker_vllm_bmm_linear.py','vllm_bmm_linear_support.py','workspace/isolated_wrappers/vllm_bmm_linear_api.py']:
   path=P/rel;code='import pathlib,sys;p=pathlib.Path('+repr(c.R+'/'+rel)+');p.write_bytes(sys.stdin.buffer.read())';subprocess.run(['ssh',c.H,'python3 -c '+shlex.quote(code)],input=path.read_bytes(),check=True)
   actual=c.remote('sha256sum '+c.R+'/'+rel,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==sha(path);records.append({'path':rel,'sha256':actual})
  c.put(P/'VLLM_BMM_LINEAR_DEPLOYMENT.json',{'checked_at':stamp(),'files':records,'CPU_AST_proof':'VLLM_BMM_LINEAR_CPU_PROOF.json','worker_proof':'VLLM_BMM_LINEAR_WORKER_PROOF.json','protocol':'both arms originalisolated3','predecessor_terminated_before_mutation':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','updated_at':stamp(),'pid':os.getpid(),'entry':'queue_vllm_bmm_linear.py','phase':'completeAPI_perbatch_matmul_vs_linear_originalisolated','durable_log':'vllm-bmm-linear-queued.log','snapshot_read_only':True})
 state.update(status='running',started_at=stamp());c.put(P/'VLLM_BMM_LINEAR_EXPANSION.json',state)
 for i in state['indices']:
  assert not stopped()
  with (P/f'bmm-linear-{i}-cached-source.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_bmm_linear_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'shared cached source qualification failed; stop instead of percase spin'
  labels=[f'rollout-vllm-bmm-linear-candidate-{i}-original-v1',f'rollout-vllm-bmm-linear-candidate-{i}-linear-v1']
  for mode,label in zip(['bmm-linear-original','bmm-linear-candidate'],labels):
   while True:
    assert not stopped()
    with (P/'controller.lock').open('a') as slot:
     fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-vllm',mode,[i],label)
    if e is not None:break
    state.update(status='resource_deferred',checked_at=stamp(),pending_label=label);c.put(P/'VLLM_BMM_LINEAR_EXPANSION.json',state);time.sleep(30)
   assert 'batch' in e,'guard/process invalid; preserve before recovery'
   r=e['batch']['results'][0]
   if r.get('phase')=='load':raise RuntimeError('shared source/load fault; stop affected group')
   if r['status']!='raw_checked_pending_guard':
    state.setdefault('failures',[]).append({'ordinal':i,'mode':mode,'checkpoint':label+'.CHECKPOINT.json'});break
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();state.update(status='running',complete_pairs=report['complete_pairs'],last_ordinal=i,checked_at=stamp());c.put(P/'VLLM_BMM_LINEAR_EXPANSION.json',state)
  if i==state['representative']:
   assert report['complete_pairs']==1,'representative failed; no expansion'
   if report['rows'][0]['device_reduction_percent']<1:
    state.update(status='representative_insufficient_gain_no_expansion',remaining_unattempted=state['indices'][1:]);break
 else:state['status']='all_exact_scope_attempted_preserve_failures'
except BaseException as exc:
 state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=stamp();c.put(P/'VLLM_BMM_LINEAR_EXPANSION.json',state)
 # Do not overwrite an active predecessor's receipt if interrupted during CPU wait.
 if not predecessor_running():c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','checked_at':stamp(),'pid':None,'prior_pid':os.getpid(),'entry':'queue_vllm_bmm_linear.py','outcome':state['status'],'next_action':'Inspect actual scope outcome and dispatch qualified successor; no blindretry','snapshot_read_only':True})
