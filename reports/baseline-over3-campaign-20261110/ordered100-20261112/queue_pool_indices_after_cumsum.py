"""Only after whole Cumsum3 terminal, apply qualified common registry entry to independent ArgMax20."""
from pathlib import Path
import json,sys,os,time,fcntl,datetime,subprocess
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';f=A/'POOL_INDICES_AFTER_CUMSUM.json';lock=(D/'pool-indices-after-cumsum.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_wait_Cumsum3_wholeformal_terminal','queued_at':now(),'scope':[208,209,210,211,337,338,339,343,344,345,346,347,348,349,350,351,354,355,356,357],'not_coverage_or_device_acceptance':True}
def put(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
put(f,s);switched=False
try:
 while True:
  assert not (D/'STOP').exists() and not (D.parent/'STOP').exists() and json.loads((D.parent/'STATE.json').read_text())['status']=='active';r=json.loads((D/'OPS_MATH_CUMSUM_EXPANSION.json').read_text());assert r['status']!='stopped_requires_diagnosis',r
  if r['status'].startswith('all3_complete'):assert r['accepted_complete_APIs']==3;break
  os.kill(r['pid'],0);time.sleep(10)
 report=json.loads((D/'OPS_MATH_CUMSUM_CURRENT.json').read_text());assert report['accepted_complete_APIs']==3 and all(x['full_checker_15raw_guard_valid'] for x in report['rows']);switched=True;put(D/'CURRENT_PROCESS.json',{'status':'active_pipeline_build_not_measurement','pid':os.getpid(),'entry':Path(__file__).name,'phase':'ArgMax20_normalregistry_diagnosis_then_boundedcompile','updated_at':now(),'durable_log':'qualified-coverage-events.jsonl'})
 for script in ['queue_pool_indices_tiling_registry_diagnosis.py','queue_math_pool_indices_compile_after_host_v3.py','assemble_math_pool_indices_h1.py']:
  assert not (D/'STOP').exists() and not (D.parent/'STOP').exists();s.update(status='foreground_'+script,checked_at=now());put(f,s)
  with (A/('POOL_INDICES_AFTER_CUMSUM-'+script+'.log')).open('w') as out:rc=subprocess.run([sys.executable,str(D/script)],stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,(script,rc)
 s.update(status='original_MaxDim_package_ready_fullAPI_qualification_required',checked_at=now());put(f,s)
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e),checked_at=now());put(f,s);raise
finally:
 if switched:put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'outcome':s['status'],'checked_at':now()})
