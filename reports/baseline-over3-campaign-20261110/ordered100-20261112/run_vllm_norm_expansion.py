"""Seven previously compile-failed exact cases, new complete API, serial guarded processes."""
import pathlib,sys,json,fcntl,time,subprocess,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'vllm-norm-expansion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert '1178' in json.loads((P/'RESOLVED_ALTERNATIVE_APIS.json').read_text())['rows']
ids=[1179,1181,1182,1193,1196,1197,1198];receipt=P/'VLLM_NORM_EXPANSION_PROCESS.json';done={}
def publish(status,i=None):
 receipt.write_text(json.dumps({'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'current':i,'indices':ids,'completed':done,'same_controller_lock':'controller.lock; no parallelNPU'},indent=2))
for i in ids:
 label='rollout-vllm-norm-candidate-'+str(i)+'-expand1'
 if (P/(label+'.FAILED.json')).exists():done[str(i)]='prior_guard_or_process_failure_preserved';continue
 while True:
  if (P/'STOP').exists() or (P.parent/'STOP').exists():publish('stopped',i);sys.exit(0)
  publish('waiting_serialized_controller_slot',i)
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);publish('attempting_fresh_guard',i);event=c.run('r-vllm','vllm-norm-expand',[i],label)
  if event is not None:break
  publish('waiting_own_host_idle_no_external_queue_wait',i);time.sleep(30)
 if 'batch' in event:
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
  done[str(i)]=event['batch']['results'][0]['status']
  subprocess.run([sys.executable,str(P/'stage_vllm_norm_recovery.py')],check=True)
 else:done[str(i)]='guard_or_process_failed_no_retry'
 publish('case_terminal',i)
publish('completed_all7_exact_cases')
