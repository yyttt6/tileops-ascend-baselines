"""Foreground CPU handoff; new guard only after exact host-generation terminal success."""
from pathlib import Path
import json,os,sys,time,datetime,subprocess,fcntl
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';f=A/'MATH_L2_SQRT_H1_BUILD_CHAIN.json';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(p,v):
 t=p.with_name(p.name+'.chain.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
lock=(D/'math-l2-sqrt-build-chain.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'host':'openlibing-DevEnv-661495','status':'CPU_wait_existing_host_generation','checked_at':now(),'scope':[373,374,375,379],'not_formal_or_coverage':True}
try:
 put(f,s);put(D/'CURRENT_PROCESS.json',{'status':'active_pipeline_build_not_measurement','pid':os.getpid(),'entry':Path(__file__).name,'phase':s['status'],'checked_at':now()})
 while True:
  assert not (D/'STOP').exists() and not (D.parent/'STOP').exists();g=json.loads((A/'SERVER1_MATH_L2_SQRT_GENERATION_V2.json').read_text())
  if 'exit_code' in g or 'error' in g:assert g.get('exit_code')==0 and g['guard']['performance_valid'],g;break
  os.kill(g['pid'],0);time.sleep(10)
 s.update(status='foreground_originalFP32compile_freshguard',checked_at=now());put(f,s)
 with (A/'math-l2-sqrt-compile-queued.log').open('a') as out:r=subprocess.run([sys.executable,str(D/'queue_server1_math_l2_sqrt_compile.py')],stdout=out,stderr=subprocess.STDOUT)
 assert r.returncode==0,r.returncode;s.update(status='originalSqrt_compiled_pending_package_and_fullAPIqualification',checked_at=now())
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e),checked_at=now());raise
finally:
 put(f,s);put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'outcome':s['status'],'checked_at':now()})
