import pathlib,sys,json,subprocess,fcntl,time,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'tla-unpack-pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);label='rollout-tla-unpack-candidate-1760-v1'
with (P/(label+'.CPU.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-tla-original prepare '+label+'-prepare 1760',stdout=out,stderr=subprocess.STDOUT).returncode
if rc:raise RuntimeError('CPU prepare rejected; no device attempt')
while True:
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-tla-original','tla-unpack',[1760],label)
 if e is not None:break
 time.sleep(30)
if 'batch' not in e:raise RuntimeError('guard invalid, preserve evidence before recovery')
subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
r=e['batch']['results'][0];(P/'TLA_UNPACK_PILOT_RESULT.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint':label+'.CHECKPOINT.json','result':r,'full_kernel_checker_precision_unchanged':True,'no_original_valid_timing_no_speedup_claim':True},indent=2))
