"""Foreground CPU wait for predecessor, then independent guarded row-scale API trial."""
import pathlib,sys,json,time,fcntl,subprocess,shlex,hashlib,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_sgl_dequant_channel import summarize
own=(P/'sgl-dequant-channel.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stopped():return (P/'STOP').exists() or (P.parent/'STOP').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
# Never mutate predecessor's bootstrap while its worker/controller is active.
while True:
 if stopped():raise RuntimeError('user STOP')
 f=P/'SGL_DEQUANT_API_EXPANSION.json'
 if f.exists() and json.loads(f.read_text()).get('status')=='all_exact_scope_attempted_preserve_failures' and subprocess.run(['ps','-p','1347958'],stdout=subprocess.DEVNULL).returncode!=0:break
 time.sleep(5)
proof=json.loads((P/'SGL_DEQUANT_CHANNEL_CPU_PROOF.json').read_text());assert proof['cases']==72 and proof['CPU_surrogate_only_not_native_acceptance']
with (P/'controller.lock').open('a') as slot:
 fcntl.flock(slot,fcntl.LOCK_EX)
 if stopped():raise RuntimeError('user STOP')
 f=P/'rollout_bootstrap.py';s=f.read_text();needle="script={'sgl-dequant-api':'worker_sgl_dequant_api.py',";assert needle in s;backup=P/'rollout_bootstrap.before-sgl-dequant-channel.py';assert not backup.exists();backup.write_text(s);f.write_text(s.replace(needle,"script={'sgl-dequant-channel':'worker_sgl_dequant_channel.py','sgl-dequant-api':'worker_sgl_dequant_api.py',",1))
 f=P/'continue_ordered.py';s=f.read_text();needle="'rollout-sgl-dequant-api-candidate-'";assert needle in s;(P/'continue_ordered.before-sgl-dequant-channel-prefix.py').write_text(s);f.write_text(s.replace(needle,needle+",'rollout-sgl-dequant-channel-candidate-'",1))
 files=['rollout_bootstrap.py','worker_sgl_dequant_channel.py','sgl_dequant_channel_support.py','workspace/isolated_wrappers/sgl_dequant_channel_api.py'];records=[]
 for rel in files:
  path=P/rel;dest=c.R+'/'+rel;code='import pathlib,sys;p=pathlib.Path('+repr(dest)+');'+('p.with_name(p.name+".before-channel").write_bytes(p.read_bytes());' if rel=='rollout_bootstrap.py' else '')+'p.write_bytes(sys.stdin.buffer.read())';subprocess.run(['ssh',c.H,'python3 -c '+shlex.quote(code)],input=path.read_bytes(),check=True);actual=c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert actual==sha(path);records.append({'path':rel,'sha256':actual})
 c.put(P/'SGL_DEQUANT_CHANNEL_DEPLOYMENT.json',{'checked_at':stamp(),'files':records,'predecessor_completed_before_bootstrap_edit':True,'CPU_proof':'SGL_DEQUANT_CHANNEL_CPU_PROOF.json','no_kernel_or_checker_changes':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','updated_at':stamp(),'pid':os.getpid(),'entry':'queue_sgl_dequant_channel.py','foreground_tool_session':'see SGL_NEXT_GROUP_QUEUE.json','phase':'SGLDequantPerChannel_completeAPI_paired_trial','durable_log':'sgl-dequant-channel-queued.log','own_migration_disabled':True,'snapshot_read_only':True})
indices=[985,983,984,981,982,980,979];state={'started_at':stamp(),'indices':indices,'representative':985,'no_predictive_memory_gate':True,'CPU_only_evidence_not_native_acceptance':True,'predecessor':'SGL_DEQUANT_API_EXPANSION.json'};c.put(P/'SGL_DEQUANT_CHANNEL_EXPANSION.json',state)
for i in indices:
 if stopped():raise RuntimeError('user STOP')
 labels=[f'rollout-sgl-dequant-channel-candidate-{i}-original-v1',f'rollout-sgl-dequant-channel-candidate-{i}-natural-v1']
 with (P/(labels[0]+'.CPU.log')).open('w') as out:rc=c.remote('python3 '+c.R+'/verify_dequant_cached_prepare.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,'CPU source/dependency failed; no percase sharedfault spin'
 for mode,label in zip(['auto','sgl-dequant-channel'],labels):
  while True:
   with (P/'controller.lock').open('a') as runlock:
    fcntl.flock(runlock,fcntl.LOCK_EX);e=c.run('r-sgl',mode,[i],label)
   if e is not None:break
   time.sleep(30)
  assert 'batch' in e,'guard/process invalid, retain evidence before any recovery'
  if e['batch']['results'][0]['status']!='raw_checked_pending_guard':
   state.setdefault('failures',[]).append({'ordinal':i,'mode':mode,'checkpoint':label+'.CHECKPOINT.json'});c.put(P/'SGL_DEQUANT_CHANNEL_EXPANSION.json',state);break
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);r=summarize();state.update({'complete_pairs':r['complete_pairs'],'last_ordinal':i,'checked_at':stamp()});c.put(P/'SGL_DEQUANT_CHANNEL_EXPANSION.json',state)
 if i==985:assert r['complete_pairs']==1,'representative notaccepted, no expansion'
state['status']='all_exact_scope_attempted_preserve_failures';c.put(P/'SGL_DEQUANT_CHANNEL_EXPANSION.json',state)
