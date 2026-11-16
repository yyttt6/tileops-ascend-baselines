"""One source-matched native pilot, then a foreground serialized controller handoff."""
import pathlib,sys,json,subprocess,fcntl,time,os,signal,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'tla-native-recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=P/'TLA_NATIVE_RECOVERY_PROCESS.json'
def publish(status,**extra):receipt.write_text(json.dumps({'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,**extra},indent=2))
label='rollout-tla-native-profile-pilot-1609-v3';publish('CPU_prepare_original_profile')
with (P/(label+'.CPU.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-tla-original prepare '+label+'-prepare 1609',stdout=out,stderr=subprocess.STDOUT).returncode
if rc:publish('CPU_prepare_failed_no_device',exit_code=rc);sys.exit(1)
while True:
 if (P/'STOP').exists() or (P.parent/'STOP').exists():publish('stopped');sys.exit(0)
 publish('waiting_serialized_controller_slot')
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);publish('fresh_guarded_original_native_pilot');event=c.run('r-tla-original','auto',[1609],label)
 if event is not None:break
 time.sleep(30)
if 'batch' not in event:publish('pilot_guard_failure_no_repeat',evidence=label+'.FAILED.json');sys.exit(1)
subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);r=event['batch']['results'][0]
(P/'TLA_NATIVE_PILOT_RESULT.json').write_text(json.dumps({'checkpoint':label+'.CHECKPOINT.json','result':r,'original_library_source_and_kernel_unchanged':True,'native_files_provenance':'TLA_ORIGINAL_NATIVE_RESTORATION.json','no_old_new_speedup_claim':True},indent=2))
if r['status']!='raw_checked_pending_guard':publish('pilot_failed_no_same_condition_retry',phase=r.get('phase'),error=r.get('error'));sys.exit(1)
with (P/'controller.lock').open('a') as slot:
 fcntl.flock(slot,fcntl.LOCK_EX);oldpid=json.loads((P/'CURRENT_PROCESS.json').read_text())['pid'];publish('pilot_accepted_waiting_safe_controller_handoff',previous_pid=oldpid)
 for _ in range(12000):
  try:w=pathlib.Path('/proc/'+str(oldpid)+'/wchan').read_text()
  except FileNotFoundError:break
  if 'lock' in w:os.kill(oldpid,signal.SIGINT);break
  time.sleep(.1)
 else:raise RuntimeError('No safe boundary; old controller left intact')
 for _ in range(100):
  if not pathlib.Path('/proc/'+str(oldpid)).exists():break
  time.sleep(.1)
 else:raise RuntimeError('Old controller not exited')
 f=P/'ROLLOUT_LEDGER.json';raw=f.read_bytes();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');f.with_name(f.name+'.before-tla-native-recovery-'+stamp).write_bytes(raw);ledger=json.loads(raw);removed=[]
 for i,e in list(ledger.items()):
  rr=e['result']
  if rr['status']=='provider_import_dependency_pending' and rr['key'][0]=='tilelang-ascend':removed.append({'ordinal':int(i),'entry':e});del ledger[i]
 tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(f)
 (P/'TLA_NATIVE_REGISTRY_REQUEUED.json').write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(removed),'old_entries':removed,'accepted_pilot_checkpoint':label+'.CHECKPOINT.json','no_valid_case_removed_or_repeated':True},indent=2))
 f=P/'CURRENT_PROCESS.json';raw=f.read_bytes();f.with_name(f.name+'.before-tla-native-recovery-'+stamp).write_bytes(raw);j=json.loads(raw);j.update(pid=os.getpid(),foreground_tool_session='see TLA_NATIVE_RECOVERY_TOOL.json',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),safe_handoff='TLA_NATIVE_RECOVERY_PROCESS.json');f.write_text(json.dumps(j,indent=2))
publish('resuming_ordered_controller_with_native_tla_profile_and_monotonic_checkpoints')
os.execv(sys.executable,[sys.executable,'-u',str(P/'continue_ordered.py')])
