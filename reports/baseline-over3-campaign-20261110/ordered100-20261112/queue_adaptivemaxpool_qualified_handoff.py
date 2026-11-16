"""Bounded dependency recovery -> full functional provenance -> formal finite4."""
from pathlib import Path
import json,datetime,time,subprocess,sys,os,fcntl
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';l=(P/'adaptivemaxpool-qualified-handoff.lock').open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'waiting_host_build_v2','counts_as_primary_dispatch':False}
def put():
 s['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=P/'ADAPTIVEMAXPOOL_QUALIFIED_HANDOFF.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
def stop():assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
def run(p,log):
 stop();s.update(status='running_child',child=str(p.name));put()
 with (P/log).open('a') as out:subprocess.run([sys.executable,'-u',str(p)],stdout=out,stderr=subprocess.STDOUT,check=True)
try:
 deadline=time.monotonic()+1200;put()
 while True:
  stop();b=json.loads((A/'SERVER2_ADAPTIVEMAXPOOL_HOST_BUILD_V2.json').read_text());assert b['status'] not in ('reference_build_failed_requires_diagnosis','stopped_requires_diagnosis'),b.get('error',b.get('exit_code'))
  if b['status']=='built_pending_identity_and_load':assert b['guard']['performance_valid'] and b['exit_code']==0;break
  assert time.monotonic()<deadline;time.sleep(10)
 run(A/'assemble_adaptivemaxpool_package.py','adaptivemaxpool-assemble.log')
 run(A/'deploy_adaptivemaxpool_package.py','adaptivemaxpool-deploy.log')
 run(P/'queue_native_adaptivemaxpool_inputs.py','native-adaptivemaxpool-inputs.log')
 run(P/'queue_server1_adaptivemaxpool_attribution.py','server1-adaptivemaxpool-attribution-queued.log')
 run(A/'qualify_adaptivemaxpool_private.py','adaptivemaxpool-qualify.log')
 run(P/'queue_ops_nn_adaptivemaxpool.py','ops-nn-adaptivemaxpool-queued.log')
 assert json.loads((P/'OPS_NN_ADAPTIVEMAXPOOL_CURRENT.json').read_text())['accepted_original_APIs']==4;s['status']='global4_allqualified_next_dispatch_required'
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:put()
