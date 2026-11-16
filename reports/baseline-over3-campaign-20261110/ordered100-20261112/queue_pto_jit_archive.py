"""Bounded foreground CPU artifact archiving for completed ownPTO attempts."""
from pathlib import Path
import json,time,os,datetime,fcntl
from archive_pto_current_jit import archive
P=Path(__file__).resolve().parent
lock=(P/'pto-jit-archive.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
state={'pid':os.getpid(),'entry':Path(__file__).name,'queued_at':now(),'status':'CPU_completed_run_archiving','no_NPU_no_newcoverage_no_deletion':True,'counts_as_primary_dispatch':False};deadline=time.monotonic()+14400
try:
 while time.monotonic()<deadline:
  if (P/'STOP').exists() or (P.parent/'STOP').exists():state['status']='stopped_user_marker';break
  for cp in sorted(P.glob('rollout-pto-current-*.CHECKPOINT.json')):archive(cp)
  records=list(P.glob('rollout-pto-current-*-JIT_ARCHIVE.json'));state.update(archives_verified=len(records),checked_at=now());f=P/'PTO_JIT_ARCHIVE_PROGRESS.json';t=f.with_suffix('.cpu.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(f)
  current=json.loads((P/'PTO_CURRENT_EXPANSION.json').read_text());pid=current.get('pid');proc=Path('/proc/'+str(pid)+'/cmdline');live=proc.exists() and b'queue_pto_current.py' in proc.read_bytes()
  if not live:state['status']='completed_all_current_closed_run_artifacts';break
  time.sleep(15)
 else:state['status']='bounded_observer_deadline_no_NPU'
except BaseException as e:state.update(status='CPU_archive_fault_preserve_requires_review',error=repr(e));raise
finally:
 state['checked_at']=now();p=P/'PTO_JIT_ARCHIVE_PROGRESS.json';t=p.with_suffix('.cpu.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
