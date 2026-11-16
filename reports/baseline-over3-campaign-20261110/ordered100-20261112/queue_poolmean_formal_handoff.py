"""Finite read-only qualification handoff; any build/probe/provenance failure stops, no numerical retry."""
from pathlib import Path
import json,datetime,fcntl,subprocess,os,time
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';lock=(P/'poolmean-formal-handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s={'entry':Path(__file__).name,'pid':os.getpid(),'queued_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'wait_original_host_build_and_guarded_functional_proof','not_formal_coverage':True}
def put():
 t=P/'POOLMEAN_FORMAL_HANDOFF.tmp';t.write_text(json.dumps(s,indent=2)+'\n');t.replace(P/'POOLMEAN_FORMAL_HANDOFF.json')
put()
try:
 while True:
  assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
  for p in [A/'SERVER2_POOLMEAN_REFERENCE_BUILD.json',P/'POOLMEAN_HOST_HANDOFF.json',A/'SERVER1_POOLMEAN_ATTRIBUTION.json']:
   if not p.exists():continue
   v=json.loads(p.read_text());status=v.get('status','');assert not any(x in status for x in ['failed','stopped_requires_diagnosis']), (p,status,v.get('error'))
  p=A/'SERVER1_POOLMEAN_ATTRIBUTION.json'
  if p.exists() and json.loads(p.read_text()).get('status')=='functional_probe_passed_not_formal_baseline':break
  time.sleep(15)
 s['status']='CPU_verify_actual_declaredSDK_identity_against_H2';put();subprocess.run(['python3',str(A/'qualify_poolmean_declared_dependency.py')],check=True)
 s['status']='exec_originalisolated15_finite12_controller';put();os.execv('/usr/bin/python3',['python3','-u',str(P/'queue_ops_nn_poolmean.py')])
except BaseException as e:s.update(status='handoff_failed_requires_diagnosis',error=repr(e),checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat());put();raise
