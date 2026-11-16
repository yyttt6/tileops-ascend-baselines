"""Sequential blockscale trial; original isolated for both complete API arms."""
import pathlib,sys,json,time,fcntl,subprocess,shlex,hashlib,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_sgl_dequant_block import summarize
f=(P/'sgl-dequant-block.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
while True:
 if stop():raise RuntimeError('user STOP')
 s=P/'SGL_DEQUANT_CHANNEL_EXPANSION.json'
 if s.exists() and json.loads(s.read_text()).get('status')=='all_exact_scope_attempted_preserve_failures' and subprocess.run(['ps','-p','1370613'],stdout=subprocess.DEVNULL).returncode!=0:break
 time.sleep(5)
assert json.loads((P/'SGL_DEQUANT_BLOCK_CPU_PROOF.json').read_text())['cases']==72
with (P/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if stop():raise RuntimeError('user STOP')
 b=P/'rollout_bootstrap.py';old=b.read_text();assert "script={'dequant-original-scoped'" in old;(P/'rollout_bootstrap.before-dequant-block-isolated.py').write_text(old);b.write_text(old.replace("script={'dequant-original-scoped'","script={'dequant-block-original':'worker_dequant_block_original_isolated.py','dequant-block-candidate':'worker_dequant_block_candidate_isolated.py','dequant-original-scoped'",1))
 b=P/'continue_ordered.py';old=b.read_text();needle="'rollout-sgl-dequant-channel-candidate-'";assert needle in old;(P/'continue_ordered.before-dequant-block-prefix.py').write_text(old);b.write_text(old.replace(needle,needle+",'rollout-sgl-dequant-block-candidate-'",1))
 names=['rollout_bootstrap.py','dequant_block_cached_fidelity.py','verify_dequant_block_cached.py','worker_dequant_block_original_isolated.py','worker_dequant_block_candidate_isolated.py','sgl_dequant_block_support.py','workspace/isolated_wrappers/sgl_dequant_block_api.py'];records=[]
 for rel in names:
  path=P/rel;code='import pathlib,sys;p=pathlib.Path('+repr(c.R+'/'+rel)+');'+('p.with_name(p.name+".before-block-isolated").write_bytes(p.read_bytes());' if rel=='rollout_bootstrap.py' else '')+'p.write_bytes(sys.stdin.buffer.read())';subprocess.run(['ssh',c.H,'python3 -c '+shlex.quote(code)],input=path.read_bytes(),check=True);actual=c.remote('sha256sum '+c.R+'/'+rel,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==sha(path);records.append({'path':rel,'sha256':actual})
 c.put(P/'SGL_DEQUANT_BLOCK_DEPLOYMENT.json',{'checked_at':stamp(),'files':records,'protocol':'ascend950-isolated-ffts-v3 botharms explicitly selected; no mixedprotocolcomparison','CPU_mapping_AST_proof':'SGL_DEQUANT_BLOCK_CPU_PROOF.json','finite_source_gate':'dequant_block_cached_fidelity.py','predecessor_finished_before_bootstrap_edit':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','updated_at':stamp(),'pid':os.getpid(),'entry':'queue_sgl_dequant_block.py','phase':'PerBlock_completeAPI_originalisolated_pairs','foreground_tool_session':'see SGL_NEXT_GROUP_QUEUE.json','durable_log':'sgl-dequant-block-queued.log','snapshot_read_only':True,'own_migration_disabled':True})
indices=[978,976,977,975,974,973];state={'started_at':stamp(),'indices':indices,'representative':978,'protocol':'originalisolated botharms','newCPUproof_not_nativeacceptance':True,'no_predictive_memory_gate':True};c.put(P/'SGL_DEQUANT_BLOCK_EXPANSION.json',state)
for i in indices:
 if stop():raise RuntimeError('user STOP')
 labels=[f'rollout-sgl-dequant-block-candidate-{i}-original-v1',f'rollout-sgl-dequant-block-candidate-{i}-natural-v1']
 with (P/(labels[0]+'.CPU.log')).open('w') as out:rc=c.remote('python3 '+c.R+'/verify_dequant_block_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,'cached metadata/source qualification rejected; stop before NPU'
 for mode,label in zip(['dequant-block-original','dequant-block-candidate'],labels):
  while True:
   with (P/'controller.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);e=c.run('r-sgl',mode,[i],label)
   if e is not None:break
   time.sleep(30)
  assert 'batch' in e,'guard/process failed, no blindretry'
  r=e['batch']['results'][0]
  if r.get('phase')=='load':raise RuntimeError('commonload/source fault; no percase spin')
  if r['status']!='raw_checked_pending_guard':
   state.setdefault('failures',[]).append({'ordinal':i,'mode':mode,'checkpoint':label+'.CHECKPOINT.json'});c.put(P/'SGL_DEQUANT_BLOCK_EXPANSION.json',state);break
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();state.update({'complete_pairs':report['complete_pairs'],'last_ordinal':i,'checked_at':stamp()});c.put(P/'SGL_DEQUANT_BLOCK_EXPANSION.json',state)
 if i==978:assert report['complete_pairs']==1,'representative notaccepted, no expansion'
state['status']='all_exact_scope_attempted_preserve_failures';c.put(P/'SGL_DEQUANT_BLOCK_EXPANSION.json',state)
