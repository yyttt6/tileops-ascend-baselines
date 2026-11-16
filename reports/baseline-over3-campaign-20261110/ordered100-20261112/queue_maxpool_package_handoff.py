"""Reuse exact existing samepin hostAPI; await independent new original devicebuild, no CPU support counts."""
from pathlib import Path
import json,datetime,subprocess,time,fcntl,os
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';lock=(P/'maxpool-package-handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s={'entry':Path(__file__).name,'pid':os.getpid(),'queued_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'wait_actual_MaxPool_devicebuild','not_formal_coverage':True}
def put():
 p=P/'MAXPOOL_PACKAGE_HANDOFF.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
put()
try:
 while True:
  assert not (P/'STOP').exists() and not (P.parent/'STOP').exists();b=json.loads((A/'SERVER2_MAXPOOL_DEVICE_BUILD.json').read_text())
  if b['status']=='built_pending_identity_and_load':assert b['exit_code']==0 and b['guard']['performance_valid'] and b['guard']['workload_exit_code']==0;break
  assert b['status'] not in ('reference_build_failed_requires_diagnosis','stopped_requires_diagnosis'),b
  time.sleep(15)
 for script in ['assemble_maxpool_package.py','deploy_maxpool_package.py']:
  s.update(status='CPU_assemble_deploy_exact_source_artifacts',script=script);put();subprocess.run(['python3',str(A/script)],check=True)
 s['status']='exec_guarded_functional_probe_after_actual_PoolMean12_terminal';put();os.execv('/usr/bin/python3',['python3','-u',str(P/'queue_server1_maxpool_attribution.py')])
except BaseException as e:s.update(status='handoff_failed_requires_diagnosis',error=repr(e),checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat());put();raise
