"""CPU-only reconnect: never reruns original build; strict original remote receipt first."""
from pathlib import Path
import subprocess,json,datetime,time,shlex,hashlib,fcntl,os,sys
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';H='openlibing-DevEnv-055409';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();statefile=A/'GEMM_PAD4_RECOVERY_CHAIN.json'
def put(p,v):
 t=p.with_suffix('.recovery.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def stop():return (D/'STOP').exists() or (D.parent/'STOP').exists()
lock=(D/'gemm-pad4-recovery-chain.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'entry':Path(__file__).name,'pid':os.getpid(),'host':H,'status':'CPU_reconnect_originalbuild_no_retry','checked_at':now(),'no_device_work_until_original_receipt_and_freshguards':True};put(statefile,s)
try:
 prior=json.loads((A/'SERVER2_GEMM_PAD4_COMPILE.json').read_text());tag=prior['tag'];receipt=N+'/'+tag+'-guard.json'
 while True:
  assert not stop();code='from pathlib import Path;import json;p=Path('+repr(receipt)+');print(json.dumps({"boot":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"guard":json.loads(p.read_text()) if p.exists() else None}))'
  try:r=subprocess.run(['ssh','-o','ConnectTimeout=8','-o','ConnectionAttempts=1',H,'/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)],capture_output=True,text=True,timeout=15);reachable=r.returncode==0
  except subprocess.TimeoutExpired:r=None;reachable=False
  if not reachable:s.update(status='CPU_H2_SSH_unreachable_originaloutcome_unknown',checked_at=now(),last_error=r.stderr[-800:] if r else 'SSH15secondtimeout',no_recompile_or_unlock=True);put(statefile,s);time.sleep(60);continue
  v=json.loads(r.stdout);g=v['guard'];assert g and g['boot_id']==v['boot'],'No matching boot-qualified receipt; originalbuild outcome unproven'
  if g['phase'] not in ('finished','released_without_launch'):
   s.update(status='CPU_originalbuild_receipt_not_terminal_no_second_submission',checked_at=now(),original_guard_phase=g['phase']);put(statefile,s);time.sleep(30);continue
  for suffix in ['-guard.json','.log']:
   remote=receipt if suffix=='-guard.json' else N+'/'+tag+'.log';r=subprocess.run(['ssh','-o','ConnectTimeout=8',H,'cat '+remote],capture_output=True,timeout=20)
   if r.returncode and suffix=='.log':continue
   assert r.returncode==0;f=A/(tag+'.RECOVERED'+suffix);assert not f.exists() or f.read_bytes()==r.stdout;f.write_bytes(r.stdout)
  assert g['phase']=='finished' and g['workload_exit_code']==0 and g['performance_valid'] is True,('Originalbuild failed; hold without blindretry',g)
  r=subprocess.run(['ssh','-o','ConnectTimeout=8',H,'cat '+N+'/PAD4_COMPILE_SELECTION.json'],capture_output=True,timeout=20,check=True);p=json.loads(r.stdout);assert p.get('completed_at') and len(p['selected_whole_commands'])==8 and p['no_flag_or_JSON_or_kernel_source_edits'];(A/'PAD4_COMPILE_SELECTION.recovered.json').write_bytes(r.stdout)
  recovery={'checked_at':now(),'original_remote_build_guard_verified_finished_exit0':True,'original_local_SSH_failure_preserved':'SERVER2_GEMM_PAD4_COMPILE.json','guard':g,'selection_sha256':hashlib.sha256(r.stdout).hexdigest(),'no_build_rerun':True};put(A/'GEMM_PAD4_REMOTE_COMPLETION_RECOVERY.json',recovery);break
 for script in ['assemble_gemm_pad4.py','stage_gemm_pad4_probe.py','queue_server1_gemm_pad4_qualification.py','audit_gemm_pad4_h1.py','queue_ops_nn_gemm_pad4.py']:
  assert not stop();s.update(status='foreground_'+script,checked_at=now());put(statefile,s)
  # The formal controller owns CURRENT_PROCESS itself, and uses fresh admission percase.
  with (D/('gemm-pad4-formal-queued.log' if script=='queue_ops_nn_gemm_pad4.py' else 'gemm-pad4-'+script+'.log')).open('a') as out:r=subprocess.run([sys.executable,str(D/script)],stdout=out,stderr=subprocess.STDOUT)
  assert r.returncode==0,(script,r.returncode)
 s.update(status='complete_originalAPI4_audit_finished',checked_at=now())
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e),checked_at=now());raise
finally:put(statefile,s)
