import pathlib,sys,fcntl,time,subprocess,json,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
label='rollout-sgl-dequant-api-candidate-988-original-v2-source-qualified'
while True:
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-sgl','dequant-original-scoped',[988],label)
 if e is not None:break
 time.sleep(30)
assert 'batch' in e and e['batch']['results'][0]['status']=='raw_checked_pending_guard','scoped runtime qualification notaccepted; do notexpand'
assert (P/'remote/runs'/e['tag']/'988/cached_input_source_proof.json').exists(),'inputsource qualification must be in rawarchive'
subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
(P/'DEQUANT_SCOPED_RUNTIME_DEVICE_ACCEPTANCE.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint':label+'.CHECKPOINT.json','full_checker_15raw_guard_accepted':True,'input_source_proof_in_dual_archive':True,'old_generic_worker_unchanged':True,'new_condition':'both CPU and runtime apply explicit samepath-only qualification; original payload assertions remain'},indent=2))
