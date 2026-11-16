"""Wait actual whole-foreground functional guard terminal, audit without another device run."""
from pathlib import Path
import json,time,os,sys,subprocess,datetime
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';f=A/'MATH_CUMSUM_QUALIFICATION_HANDOFF.json';s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_wait_qualification_terminal','not_formal_acceptance':True}
def put():
 s['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=f.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(f)
put()
try:
 while True:
  assert not (D/'STOP').exists() and not (D.parent/'STOP').exists();p=A/'SERVER1_MATH_CUMSUM_QUALIFICATION.json'
  if not p.exists():time.sleep(5);continue
  r=json.loads(p.read_text())
  if r['status']=='stopped_requires_diagnosis':raise RuntimeError(r.get('error','Probe stopped'))
  if 'exit_code' in r:assert r['exit_code']==0 and r['guard']['performance_valid'];break
  os.kill(r['pid'],0);time.sleep(10)
 with (A/'MATH_CUMSUM_QUALIFICATION_HANDOFF.log').open('w') as out:rc=subprocess.run([sys.executable,str(A/'qualify_math_cumsum.py')],stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,'Functional/provenance audit failed, no automaticretry';s['status']='qualified_pending_frozen_formal3';put()
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));put();raise
