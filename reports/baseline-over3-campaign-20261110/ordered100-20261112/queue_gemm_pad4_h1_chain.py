"""Independent H1 originaldtype variant pipeline with successful H2 recipe already established."""
from pathlib import Path
import os,sys,json,datetime,subprocess,time,fcntl
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=A/'GEMM_PAD4_H1_CHAIN.json'
def put(p,v):
 t=p.with_suffix('.chain.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def stop():return (D/'STOP').exists() or (D.parent/'STOP').exists()
lock=(D/'gemm-pad4-h1-chain.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'entry':Path(__file__).name,'pid':os.getpid(),'host':'openlibing-DevEnv-661495','status':'CPU_wait_originalcanonical_generation','checked_at':now(),'not_formal_measurement_yet':True,'H2_newpackage_identity_not_claimed':True};put(f,s);put(D/'CURRENT_PROCESS.json',{'status':'active_pipeline_build_not_measurement','pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'phase':s['status'],'snapshot_read_only':True})
try:
 while True:
  assert not stop();g=json.loads((A/'SERVER1_GEMM_PAD4_GENERATION.json').read_text())
  if 'exit_code' in g or 'error' in g:assert g.get('exit_code')==0 and g['guard']['performance_valid'],g;break
  try:os.kill(g['pid'],0)
  except ProcessLookupError:raise RuntimeError('Generation process ended without terminal receipt; no automatic restart')
  time.sleep(5)
 for script in ['queue_server1_gemm_pad4_compile.py','assemble_gemm_pad4_h1.py','stage_gemm_pad4_probe_h1.py','queue_server1_gemm_pad4_qualification.py','audit_gemm_pad4_h1.py','queue_ops_nn_gemm_pad4.py']:
  assert not stop();s.update(status='foreground_'+script,checked_at=now());put(f,s)
  if script!='queue_ops_nn_gemm_pad4.py':put(D/'CURRENT_PROCESS.json',{'status':'active_pipeline_build_or_qualification_not_formal','pid':os.getpid(),'entry':Path(__file__).name,'phase':s['status'],'checked_at':now(),'snapshot_read_only':True})
  with (D/('gemm-pad4-formal-queued.log' if script=='queue_ops_nn_gemm_pad4.py' else 'gemm-pad4-h1-'+script+'.log')).open('a') as out:r=subprocess.run([sys.executable,str(D/script)],stdout=out,stderr=subprocess.STDOUT)
  assert r.returncode==0,(script,r.returncode)
 s.update(status='complete_originaldtype4_formal_audit',checked_at=now())
except BaseException as e:
 s.update(status='stopped_requires_diagnosis',error=repr(e),checked_at=now());put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'outcome':repr(e),'checked_at':now(),'snapshot_read_only':True});raise
finally:put(f,s)
