"""Bounded deployment-only recoveries, serialized against the original controller."""
import pathlib,sys,json,fcntl,time,subprocess,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'source-recoveries.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not json.loads((P/'TORCH_FP8_PREPARE_REVISION.json').read_text())['rejected']
path=P/'RESOLVED_SOURCE_PATHS.json';resolved=json.loads(path.read_text())['rows'] if path.exists() else {};receipt=P/'SOURCE_RECOVERIES_PROCESS.json';done={}
jobs=[(i,'r-vllm','vllm-source-path','rollout-vllm-softmax-sourcepathfix-'+str(i)) for i in range(1268,1280)]+[(i,'r-template','auto','rollout-torch-fp8-sourcecache-'+str(i)) for i in range(1326,1331)]
def publish(status,i=None):
 receipt.write_text(json.dumps({'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'current':i,'scope':[x[0] for x in jobs],'completed':done,'same_controller_lock':'controller.lock, no parallelNPU'},indent=2))
for i,family,mode,label in jobs:
 if str(i) in resolved:done[str(i)]='already_accepted';continue
 if (P/(label+'.FAILED.json')).exists():done[str(i)]='prior_guard_or_process_failure_preserved';continue
 publish('CPU_prepare_source_validation',i)
 with (P/(label+'.CPU.log')).open('w') as out:
  rc=c.remote('bash '+c.R+'/launch.sh '+family+' prepare '+label+'-prepare '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
 if rc:done[str(i)]='CPU_prepare_failed_no_device_submission';continue
 while True:
  if (P/'STOP').exists() or (P.parent/'STOP').exists():publish('stopped',i);sys.exit(0)
  publish('waiting_serialized_controller_slot',i)
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);publish('attempting_fresh_guard',i);event=c.run(family,mode,[i],label)
  if event is not None:break
  publish('waiting_ownhost_idle_no_external_queue_wait',i);time.sleep(30)
 if 'batch' in event:
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);r=event['batch']['results'][0];done[str(i)]=r['status']
  if r['status']=='raw_checked_pending_guard':
   assert event['guard']['performance_valid'] and r['check']['checked_outputs']==r['check']['total_outputs']>0 and len(r['timing']['latency_samples_ms'])==15
   resolved[str(i)]={'ordinal':i,'key':r['key'],'selection':'original_complete_API_deployment_repaired','accepted_checkpoint':label+'.CHECKPOINT.json','accepted_checkpoint_sha256':c.sha(P/(label+'.CHECKPOINT.json')),'device_us':r['device_us'],'api':r['api'],'protocol':r['timing']['measurement_protocol'],'same_original_API_no_speedup_claim':True,'fix':'external source path SHA metadata' if family=='r-vllm' else 'CPU prepare source-version revision only','full_checker_15raw_guard_valid':True}
   tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':resolved,'coverage_rule':'union with original valid and alternative API resolutions; never doublecount cases; original failures retained'},indent=2));tmp.replace(path)
 else:done[str(i)]='guard_or_process_failed_no_retry'
 publish('case_terminal',i)
publish('completed_all17_bounded_source_recovery_cases')
