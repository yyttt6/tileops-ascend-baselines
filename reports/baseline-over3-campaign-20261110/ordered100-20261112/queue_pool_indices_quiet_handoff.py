"""Quiesce only this own CPU controller after full accepted checkpoint; no active device interruption."""
from pathlib import Path
import json,fcntl,datetime,time,os,signal,sys,subprocess,hashlib
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';file=A/'POOL_INDICES_QUIET_HANDOFF.json';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();own=(D/'pool-indices-quiet-handoff.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);old=json.loads((D/'OPS_MATH_POOLINDICES_EXPANSION.json').read_text());oldpid=old['pid'];s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_wait_current_wholecase_finish','old_controller_pid':oldpid,'queued_at':now(),'policy':'Do not alter active sources/env or terminate device processes. Acquire campaign controller slot, require latest full audited accepted checkpoint, signal only exact own CPU controller, preserve all accepted results, resume unaccepted with errors-only logging.','no_kernel_checker_precision_timer_raw_changes':True}
def put():
 s['checked_at']=now();tmp=file.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(file)
def stopped():return (D/'STOP').exists() or (D.parent/'STOP').exists() or json.loads((D.parent/'STATE.json').read_text())['status']!='active'
def alive(pid):
 try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
put();slot=(D/'controller.lock').open('a');held=False
try:
 while True:
  assert not stopped()
  try:fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB);held=True;break
  except BlockingIOError:time.sleep(2)
 # No campaign NPU submission can run across this local slot. Latest actual remote guard must be terminal/valid.
 cps=[]
 for p in D.glob('rollout-ops-math-pool-indices-*.CHECKPOINT.json'):
  r=json.loads(p.read_text());assert r['guard']['phase']=='finished' and r['guard']['performance_valid'] and r['guard']['workload_exit_code']==0;assert all(x['status']=='raw_checked_pending_guard' for x in r['batch']['results']),'Failure needs separate diagnosis; logging change cannot retry it';cps.append((p.stat().st_mtime,p,r))
 assert cps;_,cp,e=max(cps,key=lambda x:x[0]);target=e['indices'];deadline=time.monotonic()+90
 while True:
  assert not stopped();r=json.loads((D/'OPS_MATH_POOLINDICES_EXPANSION.json').read_text());cov=json.loads((D/'CURRENT_VALID_COVERAGE.json').read_text());audit=json.loads((D/'ROLLOUT_AUDIT.json').read_text());good=e['label'] in audit and audit[e['label']]['guard_valid'] and set(target)<=set(cov['accepted_ordinals_after100']) and r.get('last_batch')==target
  if good:break
  assert alive(oldpid) and time.monotonic()<deadline,('Not safe audited boundary',r);time.sleep(1)
 assert not (D/'MATH_POOL_INDICES_REQUIRES_DIAGNOSIS.json').exists(),'No quiet numerical retries'
 if alive(oldpid):
  args=Path('/proc/'+str(oldpid)+'/cmdline').read_bytes().split(bytes([0]));assert any(x.decode()==str((D/'queue_ops_math_pool_indices.py').resolve()) for x in args),'Only exact own oldcontroller may be interrupted'
  backup=A/('quiet-handoff-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));backup.mkdir()
  for p in [D/'OPS_MATH_POOLINDICES_EXPANSION.json',D/'CURRENT_PROCESS.json',D/'queue_ops_math_pool_indices.py',D/'worker_ops_math_pool_indices.py']:
   raw=p.read_bytes();(backup/p.name).write_bytes(raw);(backup/(p.name+'.sha256')).write_text(hashlib.sha256(raw).hexdigest())
  s.update(status='accepted_case_boundary_interrupt_own_CPU_controller',last_accepted_checkpoint=cp.name,last_accepted_guard_sha256=hashlib.sha256(json.dumps(e['guard'],sort_keys=True).encode()).hexdigest(),accepted_before_switch=r['accepted_complete_APIs']);put();os.kill(oldpid,signal.SIGINT);deadline=time.monotonic()+30
  while alive(oldpid):assert time.monotonic()<deadline,'Own CPU controller did not quiesce; no new dispatch';time.sleep(.2)
 fcntl.flock(slot,fcntl.LOCK_UN);held=False;assert not stopped();s.update(status='foreground_remaining_scope_quiet_v2',no_acceptedcase_reruns=True);put()
 with (D/'ops-math-pool-indices-quiet-v2-queued.log').open('w') as out:rc=subprocess.run([sys.executable,str(D/'queue_ops_math_pool_indices_quiet_v2.py')],stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,('Remaining quiet controller failure preserved',rc);s.update(status='scope_terminal_after_quiet_handoff');put()
except BaseException as exc:s.update(status='stopped_requires_diagnosis',error=repr(exc));put();raise
finally:
 if held:fcntl.flock(slot,fcntl.LOCK_UN)
