"""CPU handoff to a fresh guarded build after the whole current MaxPool group ends."""
from pathlib import Path
import json,os,sys,time,fcntl,datetime,subprocess
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=A/'MATH_POOL_MEAN_AFTER_MAXPOOL.json'
def put(p,v):
 t=p.with_name(p.name+'.chain.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def stopped():return (D/'STOP').exists() or (D.parent/'STOP').exists() or json.loads((D.parent/'STATE.json').read_text())['status']!='active'
lock=(D/'math-pool-mean-after-maxpool.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'entry':Path(__file__).name,'pid':os.getpid(),'host':'openlibing-DevEnv-661495','status':'CPU_wait_whole_MaxPool9_formal_terminal','queued_at':now(),'not_coverage_or_device_work':True,'next_scope':[201,202,203,204,216,217,218,219,221,222,223,224,225,227],'no_current_group_gap_insertion':True};put(f,s);switched=False
try:
 while True:
  assert not stopped();v=json.loads((D/'OPS_MATH_MAXPOOL_EXPANSION.json').read_text())
  if v['status']=='all9_complete_same_library_MaxPool_APIs_fullchecker15raw_guard_valid':assert v['accepted_complete_APIs']==9;break
  if v['status']=='stopped_requires_diagnosis':raise RuntimeError('Previousgroup needs diagnosis; do not assume quiet gap or repeat it')
  os.kill(v['pid'],0);time.sleep(10)
 assert not stopped();switched=True;s.update(status='foreground_Mean14_original_source_host_generation_freshguard',checked_at=now());put(f,s);put(D/'CURRENT_PROCESS.json',{'status':'active_pipeline_build_not_measurement','pid':os.getpid(),'entry':Path(__file__).name,'phase':s['status'],'checked_at':now(),'durable_log':'server2-alignment-20261113/math-pool-mean-generation-queued.log'})
 with (A/'math-pool-mean-generation-queued.log').open('a') as out:r=subprocess.run([sys.executable,str(D/'queue_server1_math_pool_mean_generation.py')],stdout=out,stderr=subprocess.STDOUT)
 assert r.returncode==0,r.returncode;v=json.loads((A/'SERVER1_MATH_POOL_MEAN_GENERATION.json').read_text());assert v['exit_code']==0 and v['guard']['performance_valid'];s.update(status='original_Mean_host_generated_pending_whole_FP32_command_scope_audit',checked_at=now())
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e),checked_at=now());raise
finally:
 put(f,s)
 if switched:put(D/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'outcome':s['status'],'checked_at':now()})
