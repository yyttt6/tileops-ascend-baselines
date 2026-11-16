"""Current-source paired full API trial; existing valid results not overwritten."""
import pathlib,json,sys,fcntl,time,subprocess,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'sgl-dequant-api-pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for mode,arm in [('auto','original'),('sgl-dequant-api','natural')]:
 label='rollout-sgl-dequant-api-candidate-992-'+arm+'-v1'
 if (P/'STOP').exists() or (P.parent/'STOP').exists():raise RuntimeError('user STOP')
 if mode=='auto':
  with (P/(label+'.CPU.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-sgl prepare '+label+'-prepare 992',stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'CPU dependency orpayload identity rejected; no deviceattempt'
 while True:
  with (P/'controller.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);e=c.run('r-sgl',mode,[992],label)
  if e is not None:break
  time.sleep(30)
 assert 'batch' in e,'guard/process error; preserve evidence before any recovery'
 assert e['batch']['results'][0]['status']=='raw_checked_pending_guard','candidate/original failed; no blindretry'
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
(P/'SGL_DEQUANT_API_PILOT_COMPLETE.json').write_text(json.dumps({'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ordinal':992,'labels':['rollout-sgl-dequant-api-candidate-992-original-v1','rollout-sgl-dequant-api-candidate-992-natural-v1'],'both_raw_guard_checker_valid':True,'selection_pending_source_fidelity_review':True,'no_duplicate_effective_coverage':True},indent=2))
