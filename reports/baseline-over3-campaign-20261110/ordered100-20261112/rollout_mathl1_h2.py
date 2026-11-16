"""Scoped H2 formal: official hostguard, whole-task yield, identical source/input/checker/15raw."""
from pathlib import Path
import pathlib,subprocess,json,time,datetime,hashlib,sys,shlex,tarfile,fcntl,os
P=Path(__file__).resolve().parent;ROOT=P.parent;A=P/'server2-alignment-20261113';H='openlibing-DevEnv-055409';R='/opt/baseline-over3-20261110/ordered100-20261112';STACK='/opt/baseline-over3-20261110/server2-aligned-stack';JOB=STACK+'/job-python-v1/bin/python';GUARD='/usr/local/libexec/ascend-npu/npu_guard.py';known_active={}
from nightly_window_gate import check as nightly_check
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 t=p.with_name(p.name+'.h2.tmp');t.write_text(json.dumps(x,indent=2,default=str)+'\n');t.replace(p)
def remote(cmd,**kw):return subprocess.run(['ssh','-o','ServerAliveInterval=20',H,cmd],**kw)
def tasks():
 global known_active
 result={};blocking=[]
 # A long active workflow must not disappear behind a limit of newer completed runs.
 for key,row in list(known_active.items()):
  repo=row['repo'];r=subprocess.run(['gh','run','view',str(row['databaseId']),'-R',repo,'--json','databaseId,status,conclusion,workflowName,createdAt'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
  if r.returncode:return False,{'query_failed':repo,'run':row['databaseId'],'error':r.stderr}
  current=json.loads(r.stdout)
  if current['status']=='completed':known_active.pop(key)
  else:blocking.append({'repo':repo,**current})
 if blocking:return False,{'blocking':blocking,'source':'explicit previouslyobservedrunIDs, independent of recenthistorylimit'}
 for repo in ['yyttt6/TileOPs','yyttt6/TileOPs.github.io']:
  rows=[]
  for status in ['in_progress','queued','waiting','requested','action_required']:
   r=subprocess.run(['gh','run','list','-R',repo,'--status',status,'--limit','100','--json','databaseId,status,conclusion,workflowName,createdAt'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
   if r.returncode:return False,{'query_failed':repo,'status':status,'error':r.stderr}
   rows.extend(json.loads(r.stdout))
  result[repo]=rows
  for row in rows:
   name=row['workflowName'].lower();relevant=('ascend 950' in name) if repo.endswith('/TileOPs') else ('nightly' in name)
   if relevant and row['status']!='completed':
    value={'repo':repo,**row};blocking.append(value);known_active[repo+':'+str(row['databaseId'])]=value
 return not blocking,{'blocking':blocking,'runs':result}
def admit():
 if (ROOT/'STOP').exists() or (P/'STOP').exists():raise RuntimeError('user STOP')
 n=nightly_check()
 if not n['allowed']:return False,{'nightly_window':n}
 okay,task=tasks()
 if not okay:return False,{'task_level':task}
 code='import pathlib,subprocess,shutil,json;print(json.dumps({"queue":[p.name for p in pathlib.Path("/run/lock/ascend-npu/device-0.queue").iterdir()],"workers":[s for s in subprocess.check_output(["ps","-eo","args"],text=True).splitlines() if s.split() and s.split()[0].endswith("Runner.Worker")],"disk":shutil.disk_usage("/").free}))'
 meta=json.loads(remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
 if meta['queue'] or meta['workers'] or meta['disk']<20000000000:return False,{'meta':meta,'task_level':task}
 g=remote('source '+STACK+'/environment.sh; exec '+JOB+' '+GUARD+' status --json',stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
 if g.returncode:return False,{'meta':meta,'task_level':task,'status':g.stdout,'status_error':g.stderr}
 state=json.loads(g.stdout)
 if state.get('state')!='idle' or state.get('reservation',{}).get('locked') is not False:return False,{'meta':meta,'task_level':task,'guard':state}
 okay,last=tasks();n=nightly_check();return okay and n['allowed'],{'meta':meta,'task_level':last,'guard':state,'nightly_window':n,'checked_at':now()}
def run(family,mode,indices,label):
 checkpoint=P/(label+'.CHECKPOINT.json')
 if checkpoint.exists():return json.loads(checkpoint.read_text())
 tag=label+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');okay,gate=admit();put(P/(tag+'.GATE.json'),gate)
 if not okay:put(P/'ROLLOUT_PROGRESS.json',{'status':'deferred','label':label,'host':H,'gate':gate});return None
 cmd='source '+STACK+'/environment.sh; exec '+JOB+' '+GUARD+' run --owner root/baseline-ordered100/server2/'+tag+' --wait 10 --timeout 600 --receipt '+R+'/'+tag+'-guard.json -- env BASELINE_GRAPH92_ENV_BOUND=0 '+JOB+' '+R+'/worker_ops_math_l1_h2.py '+tag+' '+' '.join(map(str,indices))
 put(P/'ROLLOUT_PROGRESS.json',{'status':'device','tag':tag,'indices':indices,'mode':mode,'family':family})
 t=time.perf_counter()
 with (P/(tag+'.log')).open('w') as out:rc=remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
 seconds=time.perf_counter()-t
 receipt=remote('cat '+R+'/'+tag+'-guard.json',stdout=subprocess.PIPE,text=True);g=json.loads(receipt.stdout) if receipt.returncode==0 else {}
 put(P/(tag+'-guard.json'),g)
 event={'label':label,'tag':tag,'indices':indices,'family':family,'mode':mode,'exit_code':rc,'guarded_seconds':seconds,'guard':g}
 with (P/'ROLLOUT_TIMING.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
 if rc==75 and (g.get('phase')=='released_without_launch' or (g.get('phase')=='waiting_lock' and not g.get('child') and 'no workload launched' in (P/(tag+'.log')).read_text())):put(P/'ROLLOUT_PROGRESS.json',{'status':'deferred_without_child','tag':tag});return None
 # Archive failures as well. Missing child/rundir is an explicit metadata-only event.
 exists=remote('test -d '+R+'/runs/'+tag).returncode==0
 if exists:
  archive=tag+'.tar.gz'
  query='import pathlib,json;p=pathlib.Path('+repr(R)+');ids='+repr(indices)+';print(json.dumps([str(pathlib.Path("prepared-rollout")/str(i)/"prepare.json") for i in ids if (p/"prepared-rollout"/str(i)/"prepare.json").is_file()]))'
  available=json.loads(remote('python3 -c '+shlex.quote(query),stdout=subprocess.PIPE,text=True,check=True).stdout)
  event['archived_prepare_metadata']=available;event['missing_prepare_metadata']=[i for i in indices if 'prepared-rollout/'+str(i)+'/prepare.json' not in available]
  remote('cd '+R+' && tar -I '+shlex.quote('gzip -1')+' -cf '+archive+' runs/'+tag+' '+tag+'-guard.json '+' '.join(available),check=True)
  with (P/archive).open('wb') as out:remote('cat '+R+'/'+archive,stdout=out,check=True)
  actual=remote('sha256sum '+R+'/'+archive,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert sha(P/archive)==actual
  with tarfile.open(P/archive) as t:t.extractall(P/'remote',filter='data')
  event['archive_sha256']=actual
 if rc or not g.get('performance_valid') or g.get('workload_exit_code')!=0 or g.get('phase')!='finished':
  put(P/(label+'.FAILED.json'),event);put(P/'ROLLOUT_PROGRESS.json',{'status':'guard_or_process_failed','event':event});return event
 batch=json.loads((P/'remote/runs'/tag/'BATCH.json').read_text());event['batch']=batch
 put(checkpoint,event);put(P/'ROLLOUT_PROGRESS.json',{'status':'batch_guard_complete_pending_local_audit','label':label,'tag':tag,'statuses':[x['status'] for x in batch['results']]})
 print('BATCH',label,seconds,[(x['ordinal'],x['status']) for x in batch['results']],flush=True)
 return event
