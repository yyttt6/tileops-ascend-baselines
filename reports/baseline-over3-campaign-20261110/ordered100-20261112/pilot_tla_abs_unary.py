import pathlib,sys,json,subprocess,fcntl,time,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'tla-unpack-pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);label='rollout-tla-abs-unary-candidate-1610-v1'
with (P/(label+'.CPU.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-tla-original prepare '+label+'-prepare 1610',stdout=out,stderr=subprocess.STDOUT).returncode
if rc:raise RuntimeError('CPU prepare rejected; no device attempt')
while True:
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-tla-original','tla-abs-unary',[1610],label)
 if e is not None:break
 time.sleep(30)
if 'batch' not in e:raise RuntimeError('guard invalid, preserve evidence before recovery')
subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
r=e['batch']['results'][0];(P/'TLA_ABS_UNARY_PILOT_RESULT.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint':label+'.CHECKPOINT.json','result':r,'full_kernel_checker_precision_unchanged':True,'original_control_available':'rollout-tla-cache-evidence-pilot-1610-v3-mapped-call.CHECKPOINT.json'},indent=2))
