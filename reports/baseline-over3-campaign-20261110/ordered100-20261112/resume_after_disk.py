import pathlib,fcntl,subprocess,sys,json,os,datetime
if (pathlib.Path(__file__).resolve().parent/'NO_OWN_MIGRATION').exists():raise SystemExit('User disabled own migration; another agent owns storage work')
P=pathlib.Path(__file__).resolve().parent
(P/'RECOVERY_PROCESS.json').write_text(json.dumps({'pid':os.getpid(),'status':'waiting_for_verified_payload_migration','next':'continue_ordered.py with global disk backoff; no individual failures caused by diskfloor','at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
with (P/'disk-recovery.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX)
if (P/'STOP').exists() or (P.parent/'STOP').exists():raise SystemExit('STOP retained')
receipt=P/'DISK_RECOVERY_RECEIPT.json';assert receipt.exists() and json.loads(receipt.read_text()),'Migration did not produce a verified receipt'
print('VERIFIED_MIGRATION_COMPLETE; resume authorized ordered campaign',flush=True)
raise SystemExit(subprocess.run([sys.executable,'-u',str(P/'continue_ordered.py')]).returncode)
