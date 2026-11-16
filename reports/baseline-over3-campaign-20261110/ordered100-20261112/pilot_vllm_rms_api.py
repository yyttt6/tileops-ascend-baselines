import pathlib,sys,json,subprocess,fcntl,time,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'tla-unpack-pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);label='rollout-vllm-rms-api-candidate-1214-v1'
with (P/(label+'.CPU.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-vllm prepare '+label+'-prepare 1214',stdout=out,stderr=subprocess.STDOUT).returncode
if rc:raise RuntimeError('CPU prepare rejected; no device attempt')
while True:
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-vllm','vllm-rms-api',[1214],label)
 if e is not None:break
 time.sleep(30)
if 'batch' not in e:raise RuntimeError('guard invalid, preserve evidence before recovery')
subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
r=e['batch']['results'][0];(P/'VLLM_RMS_API_PILOT_RESULT.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint':label+'.CHECKPOINT.json','result':r,'full_kernel_checker_precision_unchanged':True,'original_control_available':'ROLLOUT_LEDGER.json:1214'},indent=2))
