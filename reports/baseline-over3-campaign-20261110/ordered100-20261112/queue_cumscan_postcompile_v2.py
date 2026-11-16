"""CPU-only postcompile handoff; never rerun a failed/successful build."""
from pathlib import Path
import json,os,sys,time,datetime,subprocess,fcntl
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';f=A/'MATH_CUMSCAN_POSTCOMPILE_V2.json';lock=(D/'cumscan-postcompile-v2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_wait_original_compile_terminal','checked_at':now(),'not_coverage_or_device_work':True}
def put():
 t=f.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(f)
put()
try:
 while True:
  assert not (D/'STOP').exists() and not (D.parent/'STOP').exists() and json.loads((D.parent/'STATE.json').read_text())['status']=='active';r=json.loads((A/'MATH_CUMSCAN_COMPILE_AFTER_HOST_V2.json').read_text())
  if r['status']=='stopped_requires_diagnosis':raise RuntimeError('Originalbuild failure preserved: '+r.get('error',''))
  if r['status']=='original_all_float_commands_compiled_pending_package_fullAPIqualification':break
  os.kill(r['pid'],0);time.sleep(10)
 s.update(status='CPU_package_verified_compiled_artifacts',checked_at=now());put()
 with (A/'MATH_CUMSCAN_POSTCOMPILE_V2.log').open('w') as out:rc=subprocess.run([sys.executable,str(D/'assemble_math_cumsum_h1.py')],stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,'CPU package failed; no device or blindretry'
 s.update(status='package_ready_full_API_device_qualification_required',checked_at=now());put()
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e),checked_at=now());put();raise
