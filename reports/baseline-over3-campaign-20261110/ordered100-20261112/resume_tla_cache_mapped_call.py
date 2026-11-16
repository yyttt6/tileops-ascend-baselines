import pathlib,json,sys,subprocess,fcntl,time,datetime,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'jit-cache-recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(n,v):(P/n).write_text(json.dumps(v,indent=2))
label='rollout-tla-cache-evidence-pilot-1610-v3-mapped-call'
with (P/(label+'.CPU.log')).open('w') as out:
 rc=c.remote('bash '+c.R+'/launch.sh r-tla-original prepare '+label+'-prepare 1610',stdout=out,stderr=subprocess.STDOUT).returncode
if rc:raise RuntimeError('CPU prepare failed')
while True:
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-tla-original','auto',[1610],label)
 if e is not None and 'batch' not in e and e.get('guard',{}).get('phase')=='released_without_launch':
  old=P/(label+'.FAILED.json')
  if old.exists():old.rename(P/(label+'.admission-'+str(time.time_ns())+'.FAILED.json'))
  e=None
 if e is not None and 'batch' not in e and e.get('guard',{}).get('visibility_lost') and e.get('guard',{}).get('workload_exit_code')==0:
  bf=P/'remote/runs'/e['tag']/'BATCH.json'
  if bf.exists() and all(r['status']=='raw_checked_pending_guard' for r in json.loads(bf.read_text())['results']):
   old=P/(label+'.FAILED.json')
   if old.exists():old.rename(P/(label+'.visibility-invalid-'+str(time.time_ns())+'.FAILED.json'))
   put('TLA_CACHE_GUARD_BACKOFF.json',{'checked_at':stamp(),'event':e,'action':'raw/checker success but guard observation timeout: no acceptance; await fresh guard, preserve artifacts; retry only observation-invalid exact pilot'})
   e=None
 if e is not None:break
 time.sleep(30)
if 'batch' not in e:raise RuntimeError('pilot guard failure')
subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
if e['batch']['results'][0]['status']!='raw_checked_pending_guard':raise RuntimeError('pilot failed; no automatic same-condition retry')
f=P/'ROLLOUT_LEDGER.json';raw=f.read_bytes();ts=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');f.with_name(f.name+'.before-cache-evidence-'+ts).write_bytes(raw);j=json.loads(raw);removed=[]
for k,v in list(j.items()):
 r=v.get('result',{});error=str(r)
 if r.get('status')!='raw_checked_pending_guard' and r.get('key',[None])[0]=='tilelang-ascend' and 'NoneType' in error and ('adapter.libpath' in error or 'jit_evidence.py' in error):removed.append(v);del j[k]
sf=P/'PINNED_DEPENDENCY_SUPERSEDED.json';b=sf.read_bytes();sf.with_name(sf.name+'.before-cache-evidence-'+ts).write_bytes(b);s=json.loads(b);s['checkpoint_labels']=sorted(set(s['checkpoint_labels'])|{v['label'] for v in removed});sf.write_text(json.dumps(s,indent=2));f.write_text(json.dumps(j,indent=2))
f=P/'continue_ordered.py';b=f.read_bytes();f.with_name(f.name+'.before-cache-evidence-'+ts).write_bytes(b);txt=b.decode();needle=";publish('cpu_preparation',ids)";assert txt.count(needle)==1;txt=txt.replace(needle,"+('-cache-evidence2' if row['library']=='tilelang-ascend' else '')"+needle);compile(txt,str(f),'exec');f.write_text(txt)
put('TLA_CACHE_EVIDENCE_RECOVERY.json',{'checked_at':stamp(),'pilot_checkpoint':label+'.CHECKPOINT.json','removed_failed_entries':removed,'no_valid_results_removed':True,'math_checker_kernel_unchanged':True})
put('CURRENT_PROCESS.json',{'pid':os.getpid(),'updated_at':stamp(),'entry':'continue_ordered.py','foreground_tool_session':'see TLA_CACHE_EVIDENCE_TOOL.json','own_migration_disabled':True})
os.execv(sys.executable,[sys.executable,'-u',str(P/'continue_ordered.py')])
