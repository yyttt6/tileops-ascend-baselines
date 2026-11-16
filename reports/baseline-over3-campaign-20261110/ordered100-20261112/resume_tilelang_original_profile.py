"""Two exact pilot processes, then resume only libraries with accepted device proof."""
import pathlib,sys,json,subprocess,fcntl,time,os,datetime
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'tilelang-registration-recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old_process=json.loads((P/'CURRENT_PROCESS.json').read_text());assert not pathlib.Path('/proc/'+str(old_process['pid'])).exists(),'original controller still running'
assert all(v['imported'] for v in json.loads((P/'TILELANG_CPU_REGISTRY_PREFLIGHT.json').read_text())['libraries'].values())
receipt=P/'TILELANG_REGISTRATION_PROCESS.json';out={};passed=[]
def publish(status):receipt.write_text(json.dumps({'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'pilots':out,'passed_libraries':passed},indent=2))
for i,lib in [(1546,'tilelang')]:
 label='rollout-tilelang-registration-native-v2-'+str(i);publish('CPU_prepare_'+lib)
 with (P/(label+'.CPU.log')).open('w') as log:rc=c.remote('bash '+c.R+'/launch.sh r-template prepare '+label+'-prepare '+str(i),stdout=log,stderr=subprocess.STDOUT).returncode
 if rc:out[lib]={'phase':'CPU_prepare','exit_code':rc,'log':label+'.CPU.log'};continue
 if (P/(label+'.FAILED.json')).exists():out[lib]={'phase':'prior_guard_failure_preserved','evidence':label+'.FAILED.json'};continue
 while True:
  if (P/'STOP').exists() or (P.parent/'STOP').exists():publish('stopped');sys.exit(0)
  publish('pilot_'+lib)
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);event=c.run('r-template','auto',[i],label)
  if event is not None:break
  time.sleep(30)
 if 'batch' in event:
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);r=event['batch']['results'][0];out[lib]={'phase':r.get('phase'),'status':r['status'],'evidence':label+'.CHECKPOINT.json','error':r.get('error')}
  if r['status']=='raw_checked_pending_guard':passed.append(lib)
 else:out[lib]={'phase':'guard_or_process_failed','evidence':label+'.FAILED.json'}
 publish('pilot_terminal')
if not passed:publish('pilots_require_dependency_review_no_automatic_repeat');sys.exit(1)
f=P/'ROLLOUT_LEDGER.json';raw=f.read_bytes();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');f.with_name(f.name+'.before-tilelang-recovery-'+stamp).write_bytes(raw);ledger=json.loads(raw);removed=[]
for i,e in list(ledger.items()):
 r=e['result']
 if r['status']=='provider_import_dependency_pending' and r['key'][0] in passed:removed.append({'ordinal':int(i),'entry':e});del ledger[i]
tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(f)
(P/'TILELANG_REGISTRATION_REQUEUED.json').write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed_libraries':passed,'requeued_count':len(removed),'old_deferred_entries':removed,'pilots':out,'old_failures_preserved':True},indent=2));publish('resuming_ordered_controller_after_verified_registry_repair')
f=P/'CURRENT_PROCESS.json';old=f.read_bytes();f.with_name(f.name+'.before-tilelang-recovery-'+stamp).write_bytes(old);j=json.loads(old);j.update(pid=os.getpid(),foreground_tool_session='see TILELANG_REGISTRATION_TOOL.json',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),safe_handoff='TILELANG_REGISTRATION_PROCESS.json');f.write_text(json.dumps(j,indent=2))
os.execv(sys.executable,[sys.executable,'-u',str(P/'continue_ordered.py')])
