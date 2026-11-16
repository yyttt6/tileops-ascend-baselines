"""Finish the active guarded batch, then pause only this controller for evidence repair."""
import pathlib,json,fcntl,os,signal,time,datetime
P=pathlib.Path(__file__).resolve().parent;pid=json.loads((P/'CURRENT_PROCESS.json').read_text())['pid'];f=P/'JIT_CACHE_EVIDENCE_REPAIR_PROCESS.json'
def put(status):f.write_text(json.dumps({'status':status,'previous_controller_pid':pid,'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
put('waiting_active_guard_archive_safe_boundary')
with (P/'controller.lock').open('a') as slot:
 fcntl.flock(slot,fcntl.LOCK_EX)
 for _ in range(12000):
  try:w=pathlib.Path('/proc/'+str(pid)+'/wchan').read_text()
  except FileNotFoundError:break
  if 'lock' in w:os.kill(pid,signal.SIGINT);break
  time.sleep(.1)
 else:raise RuntimeError('No safe boundary; no forced NPU interruption')
 for _ in range(100):
  if not pathlib.Path('/proc/'+str(pid)).exists():break
  time.sleep(.1)
 else:raise RuntimeError('Controller did not exit')
put('controller_safely_paused_CPU_evidence_repair_in_progress')
