#!/usr/bin/env python3
"""CPU supervisor; every foreground device batch owns one bounded npu-guard."""
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
from specs import ROOT,EVIDENCE,CACHE
p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--tag',required=True);p.add_argument('--batch-size',type=int,default=6);p.add_argument('--max-batches',type=int);p.add_argument('--compile-batch',action='store_true',help='Allow 900 seconds only for original graph JIT compilation batches');p.add_argument('--start',type=int,default=0);p.add_argument('--poll',type=float,default=1.0);a=p.parse_args()
jobs=json.loads(a.jobs.read_text());script=Path(__file__).with_name('run_cases.py');state=EVIDENCE/'progress.json';own=Path(__file__).parent
for n,start in enumerate(range(a.start,len(jobs),a.batch_size)):
 if a.max_batches is not None and n>=a.max_batches:break
 name=f'{a.tag}_{start:04d}';run=EVIDENCE/'runs'/name
 if run.exists():raise FileExistsError(run)
 run.mkdir(parents=True);selected=jobs[start:start+a.batch_size];jobfile=run/'jobs.json';jobfile.write_text(json.dumps(selected,indent=2))
 shell='source '+str(CACHE/'graph-runtime/vendors/allpairs_math_math/bin/set_env.bash')+'\nexport PYTHONDONTWRITEBYTECODE=1\nexec python '+str(script.resolve())+' --jobs '+str(jobfile)+' --output-dir '+str(run)
 cmd=['npu-guard','run','--owner','allpairs/ops-math-continuation','--wait','7200','--poll',str(a.poll),'--timeout','900' if a.compile_batch else '300','--',str(ROOT/'.tools/baselines/baseline-run'),'ops-math','bash','-c',shell]
 data={'phase':'device_batch_queued_or_running','owner':'allpairs/ops-math-continuation','tag':a.tag,'start':start,'batch_jobs':selected,'command':cmd,'compile_batch':a.compile_batch,'soft_budget_seconds':60,'timeout_reason':'Original per-manifest-shape CANN graph JIT compilation' if a.compile_batch else 'Bounded runtime correctness checks','updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'coverage':'coverage.json'};state.write_text(json.dumps(data,indent=2));(run/'command.json').write_text(json.dumps(data,indent=2))
 with (run/'guard.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 (run/'guard_exit.json').write_text(json.dumps({'exit_code':r.returncode,'ended_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
 print(json.dumps({'run':name,'guard_exit':r.returncode}),flush=True)
 if (run/'native-events.jsonl').exists():subprocess.run([sys.executable,str(own/'accept.py'),str(run)],check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 subprocess.run([sys.executable,str(own/'report.py')],check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 subprocess.run([sys.executable,str(own/'publish.py')],check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 if r.returncode!=0:
  data.update(phase='batch_guard_stopped',guard_exit=r.returncode);state.write_text(json.dumps(data,indent=2));sys.exit(r.returncode)
 summary=run/'batch_summary.json'
 if summary.exists() and json.loads(summary.read_text()).get('remaining_jobs'):
  data.update(phase='partial_batch_complete',remaining_jobs=json.loads(summary.read_text())['remaining_jobs']);state.write_text(json.dumps(data,indent=2));break
 data.update(phase='batch_complete',next_start=start+len(selected));state.write_text(json.dumps(data,indent=2))
 if (EVIDENCE/'STOP_AFTER_BATCH').exists():break
else:
 data.update(phase='queue_complete');state.write_text(json.dumps(data,indent=2))
