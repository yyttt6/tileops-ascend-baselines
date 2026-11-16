"""Necessary contemporaneous original control after reference runtime relocation."""
import pathlib,sys,json,fcntl,time,subprocess,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
label='rollout-vllm-rms-api-candidate-1214-original-control-private-reference-v1'
while True:
 with (P/'controller.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);e=c.run('r-vllm','auto',[1214],label)
 if e is not None:break
 time.sleep(30)
assert 'batch' in e,'guard or process failed; no blind retry'
subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
(P/'VLLM_RMS_PRIVATE_CURRENT_CONTROL.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint':label+'.CHECKPOINT.json','result':e['batch']['results'][0],'reason':'original historical fulloverlay tree identity not recorded; fresh original control in exactly same restoredprivateTorch210 environment as candidate','not_duplicate_coverage':True},indent=2))
