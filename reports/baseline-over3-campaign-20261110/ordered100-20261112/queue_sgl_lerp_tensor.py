"""Bounded complete-API trial, qualified before dispatch; fail closed on shared faults."""
import pathlib,sys,json,time,fcntl,subprocess,shlex,hashlib,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_sgl_lerp_tensor import summarize
own=(P/'sgl-lerp-tensor.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state={'started_at':stamp(),'indices':[1017,1016,1020,1018,1019],'representative':1017,'complete_pairs':0,'status':'preparing','no_predictive_memory_gate':True,'no_kernel_checker_precision_changes':True}
try:
 assert not stopped()
 assert json.loads((P/'SGL_DEQUANT_BLOCK_EXPANSION.json').read_text())['complete_pairs']==6
 assert json.loads((P/'SGL_LERP_TENSOR_CPU_PROOF.json').read_text())['cases']==48
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  b=P/'rollout_bootstrap.py';old=b.read_text();new=old
  if "'lerp-tensor-original':" not in old:new=old.replace("script={","script={'lerp-tensor-original':'worker_lerp_tensor_original.py','lerp-tensor-pad':'worker_sgl_lerp_tensor_pad.py',",1)
  b.with_name('rollout_bootstrap.before-lerp-tensor.py').write_text(old);b.write_text(new)
  b=P/'continue_ordered.py';old=b.read_text();needle="'rollout-sgl-dequant-block-candidate-'";assert needle in old
  if "'rollout-sgl-lerp-tensor-candidate-'" not in old:b.with_name('continue_ordered.before-lerp-tensor.py').write_text(old);b.write_text(old.replace(needle,needle+",'rollout-sgl-lerp-tensor-candidate-'",1))
  records=[]
  for rel in ['rollout_bootstrap.py','lerp_tensor_cached_fidelity.py','verify_lerp_tensor_cached.py','worker_lerp_tensor_original.py','worker_sgl_lerp_tensor_pad.py','sgl_lerp_tensor_support.py','workspace/isolated_wrappers/sgl_lerp_tensor_pad_api.py']:
   path=P/rel;code='import pathlib,sys;p=pathlib.Path('+repr(c.R+'/'+rel)+');p.write_bytes(sys.stdin.buffer.read())'
   subprocess.run(['ssh',c.H,'python3 -c '+shlex.quote(code)],input=path.read_bytes(),check=True)
   actual=c.remote('sha256sum '+c.R+'/'+rel,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==sha(path);records.append({'path':rel,'sha256':actual})
  c.put(P/'SGL_LERP_TENSOR_DEPLOYMENT.json',{'checked_at':stamp(),'files':records,'CPU_AST_proof':'SGL_LERP_TENSOR_CPU_PROOF.json','worker_proof':'SGL_LERP_TENSOR_WORKER_PROOF.json','protocol':'both arms originalisolated3'})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','updated_at':stamp(),'pid':os.getpid(),'entry':'queue_sgl_lerp_tensor.py','phase':'completeAPI_qualified_originalisolated_pairs','durable_log':'sgl-lerp-tensor-queued.log','snapshot_read_only':True})
 state['status']='running';c.put(P/'SGL_LERP_TENSOR_EXPANSION.json',state)
 for i in state['indices']:
  assert not stopped()
  with (P/f'lerp-tensor-{i}-cached-source.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_lerp_tensor_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'shared cached source/dependency qualification failed; no per-case retries'
  labels=[f'rollout-sgl-lerp-tensor-candidate-{i}-original-v1',f'rollout-sgl-lerp-tensor-candidate-{i}-pad-v1']
  for mode,label in zip(['lerp-tensor-original','lerp-tensor-pad'],labels):
   while True:
    assert not stopped()
    with (P/'controller.lock').open('a') as slot:
     fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-sgl',mode,[i],label)
    if e is not None:break
    state.update(status='resource_deferred',checked_at=stamp(),pending_label=label);c.put(P/'SGL_LERP_TENSOR_EXPANSION.json',state);time.sleep(30)
   assert 'batch' in e,'guard/process failure; preserve before recovery'
   r=e['batch']['results'][0]
   if r.get('phase')=='load':raise RuntimeError('shared source/load fault; stop affected group')
   if r['status']!='raw_checked_pending_guard':
    state.setdefault('failures',[]).append({'ordinal':i,'mode':mode,'checkpoint':label+'.CHECKPOINT.json'});break
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();state.update(status='running',complete_pairs=report['complete_pairs'],checked_at=stamp(),last_ordinal=i);c.put(P/'SGL_LERP_TENSOR_EXPANSION.json',state)
  if i==state['representative']:
   assert report['complete_pairs']==1,'representative failed; no expansion'
   if report['rows'][0]['device_reduction_percent']<1:
    state.update(status='representative_insufficient_gain_no_expansion',remaining_unattempted=state['indices'][1:]);break
 else:state['status']='all_exact_scope_attempted_preserve_failures'
except BaseException as exc:
 state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=stamp();c.put(P/'SGL_LERP_TENSOR_EXPANSION.json',state)
 c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','checked_at':stamp(),'pid':None,'prior_pid':os.getpid(),'entry':'queue_sgl_lerp_tensor.py','outcome':state['status'],'next_action':'Inspect outcome, choose qualified next scope; no blind retry or stale active state','snapshot_read_only':True})
