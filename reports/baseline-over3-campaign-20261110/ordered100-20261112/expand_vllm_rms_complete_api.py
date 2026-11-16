"""Exact frozen RMS cases, supported complete API, fresh current-runtime paired controls."""
import pathlib,json,sys,math,time,fcntl,subprocess,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_vllm_rms_complete_api import summarize
lock=(P/'vllm-rms-complete-expansion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert summarize()['complete_pairs']==1,'representative proof required; use checkpoint resume rather than duplicated launch'
rows=json.loads((P/'TASKS.json').read_text());eligible=[];unsupported=[]
for i in range(1215,1246):
 m=json.loads((P/'remote/prepared-rollout'/str(i)/'prepare.json').read_text());par=m['params'];hidden=math.prod(par.get('normalized_shape',[par['x'][0][-1]]));size={'float16':2,'bfloat16':2,'float32':4}[par['x'][1]]
 (eligible if hidden*size<=65536 else unsupported).append(i)
state={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'eligible':eligible,'static_feature_limit_original_retained':unsupported,'representative_completed':1214,'no_predictive_memory_gate':True,'current_original_controls_reason':'full oldoverlay tree hashes not recorded; all pairedcontrols use restoredmatchedTorch210 exactsameprivate runtime','device_metric_only_selection':True,'historical_ratios_unchanged':True}
c.put(P/'VLLM_RMS_COMPLETE_API_EXPANSION.json',state)
for i in eligible:
 if (P/'STOP').exists() or (P.parent/'STOP').exists():raise RuntimeError('user STOP')
 labels=[f'rollout-vllm-rms-api-candidate-{i}-original-private-v1',f'rollout-vllm-rms-api-candidate-{i}-alternate-private-v1']
 with (P/(labels[0]+'.CPU.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-vllm prepare '+labels[0]+'-prepare '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,'CPU commondependency orprepareidentity fault: stop group rather than percase spin'
 for mode,label in zip(['auto','vllm-rms-api'],labels):
  while True:
   with (P/'controller.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX);e=c.run('r-vllm',mode,[i],label)
   if e is not None:break
   time.sleep(30)
  assert 'batch' in e,'bootstrap/guard failed: retain evidence and stop affected group'
  if e['batch']['results'][0].get('status')!='raw_checked_pending_guard':
   state.setdefault('individual_failures',[]).append({'ordinal':i,'mode':mode,'checkpoint':label+'.CHECKPOINT.json'});c.put(P/'VLLM_RMS_COMPLETE_API_EXPANSION.json',state);break
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
 r=summarize();state['complete_pairs']=r['complete_pairs'];state['last_checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();state['last_ordinal']=i;c.put(P/'VLLM_RMS_COMPLETE_API_EXPANSION.json',state)
state['status']='all_eligible_attempted_preserve_individual_failures';c.put(P/'VLLM_RMS_COMPLETE_API_EXPANSION.json',state)
