"""Foreground, same-host safe-boundary controller handoff; no background escape."""
import pathlib,os,time,json,fcntl,signal,datetime,sys
P=pathlib.Path(__file__).resolve().parent;old_pid=945423;receipt=P/'CPU_CASE_HANDOFF.json'
def put(status,**extra):receipt.write_text(json.dumps({'status':status,'pid':os.getpid(),'previous_pid':old_pid,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra},indent=2))
put('waiting_current_guard_and_archive_to_finish')
with (P/'controller.lock').open('a') as slot:
 fcntl.flock(slot,fcntl.LOCK_EX)
 put('waiting_next_device_submission_boundary')
 for _ in range(12000):
  try:w=pathlib.Path('/proc/'+str(old_pid)+'/wchan').read_text()
  except FileNotFoundError:break
  if 'lock' in w:os.kill(old_pid,signal.SIGINT);break
  time.sleep(.1)
 else:put('safe_boundary_timeout_no_interruption');raise RuntimeError('No safe handoff')
 for _ in range(100):
  if not pathlib.Path('/proc/'+str(old_pid)).exists():break
  time.sleep(.1)
 else:raise RuntimeError('Old controller not exited')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');f=P/'ROLLOUT_LEDGER.json';raw=f.read_bytes();f.with_name(f.name+'.before-exactcase-handoff-'+stamp).write_bytes(raw);ledger=json.loads(raw);removed=[]
 for i in range(1497,1503):
  e=ledger[str(i)];r=e['result'];assert r['status']=='provider_dependency_deferred' and r['cause']['phase']=='CPU_prepare'
  removed.append({'ordinal':i,'entry':e});del ledger[str(i)]
 tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(f)
 f=P/'CPU_PARTIAL_BATCH_ATTRIBUTION_REPAIR.json';j=json.loads(f.read_text());j.update(previous_deferred_entries=removed,activated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='six CPU-only deferred cases requeued; only absent exact IDs will remain held');f.write_text(json.dumps(j,indent=2))
 f=P/'CURRENT_PROCESS.json';raw=f.read_bytes();f.with_name(f.name+'.before-exactcase-handoff-'+stamp).write_bytes(raw);j=json.loads(raw);j.update(pid=os.getpid(),foreground_tool_session='see CPU_CASE_HANDOFF_TOOL.json',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),safe_handoff='CPU_CASE_HANDOFF.json');f.write_text(json.dumps(j,indent=2))
put('resuming_ordered_controller_with_exact_case_failure_attribution')
os.execv(sys.executable,[sys.executable,'-u',str(P/'continue_ordered.py')])
