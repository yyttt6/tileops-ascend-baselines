import pathlib,json,fcntl,subprocess,sys,datetime,os
P=pathlib.Path(__file__).resolve().parent
with (P/'continuation.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX)
# Only this turn's internal drain marker may be removed, never a user STOP.
s=P/'STOP';x=json.loads(s.read_text());assert x.get('internal_control_hold') and x.get('reason','').startswith('slowcase140s') and not (P.parent/'STOP').exists()
s.rename(P/'STOP.internal-slowbatch-drain.resolved.json')
code="import pathlib,json;p=pathlib.Path('/opt/baseline-over3-20261110/ordered100-20261112/STOP');x=json.loads(p.read_text());assert x.get('internal_control_hold') and x.get('reason','').startswith('drain at nextcase');p.rename(p.with_name('STOP.internal-slowbatch-drain.resolved.json'))"
subprocess.run(['ssh','openlibing-DevEnv-661495','python3 -c '+__import__('shlex').quote(code)],check=True)
ledger_path=P/'ROLLOUT_LEDGER.json';raw=ledger_path.read_bytes();ledger=json.loads(raw);removed={}
for i in range(690,698):
 key=str(i);e=ledger.get(key,{})
 if e.get('result',{}).get('status')=='guard_or_process_invalid_pending' and e['result'].get('evidence')=='rollout-690-697.FAILED.json':removed[key]=ledger.pop(key)
(P/'ROLLOUT_LEDGER.before-slowbatch-requeue.json').write_bytes(raw);ledger_path.write_text(json.dumps(ledger,indent=2));(P/'SLOW_BATCH_REQUEUED.json').write_text(json.dumps({'invalid_guard_results_not_accepted':True,'new_condition':'singlecase per600s guard instead of8slowcases','cases':removed},indent=2))
child=subprocess.Popen([sys.executable,'-u',str(P/'continue_ordered.py')]);(P/'CURRENT_PROCESS.json').write_text(json.dumps({'pid':child.pid,'supervisor_pid':os.getpid(),'entry':'continue_ordered.py','updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'own_migration_disabled':True},indent=2));raise SystemExit(child.wait())
