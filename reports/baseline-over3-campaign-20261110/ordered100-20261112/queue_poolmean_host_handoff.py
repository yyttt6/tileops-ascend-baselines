"""CPU successor waits actual reference build; host API package and guarded probe, no false formal count."""
from pathlib import Path
import json,time,datetime,os,subprocess,fcntl
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';lock=(P/'poolmean-host-handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def put(x):
 p=P/'POOLMEAN_HOST_HANDOFF.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
s={'entry':Path(__file__).name,'pid':os.getpid(),'queued_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'wait_actual_H2_original_host_API_build','not_formal_coverage':True};put(s)
try:
 while True:
  assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
  b=json.loads((A/'SERVER2_POOLMEAN_REFERENCE_BUILD.json').read_text())
  if b['status']=='built_pending_identity_and_load':
   assert b['exit_code']==0 and b['guard']['performance_valid'] is True and b['guard']['workload_exit_code']==0;break
  assert b['status'] not in ('stopped_requires_diagnosis','reference_build_failed_requires_diagnosis'),b
  time.sleep(15)
 for script in ['/tmp/assemble_poolmean_host_package.py','/tmp/deploy_poolmean_host_package.py']:
  s.update(status='CPU_original_API_package_assembly_and_SHA_deployment',script=script);put(s);subprocess.run(['python3',script],check=True)
 s.update(status='exec_guarded_functional_probe_not_formal',checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat());put(s)
 os.execv('/usr/bin/python3',['python3','-u',str(P/'queue_server1_poolmean_attribution.py')])
except BaseException as e:s.update(status='handoff_failed_requires_diagnosis',error=repr(e),checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat());put(s);raise
