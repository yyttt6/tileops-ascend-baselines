"""Wait for independent fresh valid functional probe; never accepts build timing."""
from pathlib import Path
import json,datetime,time,subprocess,sys,os,fcntl,hashlib,shlex
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-fp8quant-v1';l=(P/'fp8quant-qualified-handoff.lock').open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'waiting_independent_functional_probe','counts_as_primary_dispatch':False,'original_build_guard_valid':True}
def put():
 s['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=P/'FP8QUANT_QUALIFIED_HANDOFF.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
def stop():assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
def run(p,log):
 stop();s.update(status='running_child',child=p.name);put()
 with (P/log).open('a') as out:subprocess.run([sys.executable,'-u',str(p)],stdout=out,stderr=subprocess.STDOUT,check=True)
try:
 deadline=time.monotonic()+1200;put()
 while True:
  stop();b=json.loads((A/'SERVER1_FP8QUANT_ATTRIBUTION.json').read_text());status=b['status'];assert status not in ('functional_probe_failed_requires_diagnosis','stopped_requires_diagnosis'),b.get('error',b.get('exit_code'))
  if status=='functional_probe_passed_not_formal_baseline':assert b['guard']['performance_valid'] is True and b['guard']['workload_exit_code']==0 and b['exit_code']==0;break
  assert time.monotonic()<deadline;time.sleep(5)
 # Original controller archived6 probe groups; preserve all8 without altering its source/run.
 proof={}
 for i in range(6):
  name=b['result']+'.group-'+str(i)+'.pt';path=N+'/'+name;raw=subprocess.check_output(['ssh','openlibing-DevEnv-661495','cat '+shlex.quote(path)]);sha=hashlib.sha256(raw).hexdigest();assert subprocess.check_output(['ssh','openlibing-DevEnv-661495','sha256sum '+shlex.quote(path)],text=True).split()[0]==sha;p=A/name;assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw);proof[name]=sha
 (A/'FP8QUANT_ALL6_FUNCTIONAL_GROUP_ARCHIVE.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_SHA256':proof,'fresh_functional_guard_valid':True,'original_build_guard_valid':True},indent=2)+'\n')
 run(A/'qualify_fp8quant_private.py','fp8quant-qualify.log')
 assert json.loads((P/'OPS_NN_FP8QUANT_INPUTS.json').read_text())['verified_inputs']==3
 run(P/'queue_ops_nn_fp8quant.py','ops-nn-fp8quant-queued.log')
 assert json.loads((P/'OPS_NN_FP8QUANT_CURRENT.json').read_text())['accepted_original_APIs']==3;s['status']='fp8quant3_allqualified_next_dispatch_required'
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:put()
