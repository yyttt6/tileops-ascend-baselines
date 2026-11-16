"""Foreground bounded batch driver; no child or numerical retry behind a guard."""
import pathlib,subprocess,json,time,datetime,hashlib,sys,shlex,tarfile,fcntl,os
P=pathlib.Path(__file__).resolve().parent;ROOT=P.parent;H='openlibing-DevEnv-661495';R='/opt/baseline-over3-20261110/ordered100-20261112'
ENV='env LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64 ';PY='/opt/tileops-ci/runtime/bin/python -I -S '
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.write_text(json.dumps(x,indent=2,default=str))
def remote(cmd,**kw):return subprocess.run(['ssh','-o','ServerAliveInterval=20',H,cmd],**kw)
def admit():
 if (ROOT/'STOP').exists() or (P/'STOP').exists():raise RuntimeError('user STOP')
 code='import pathlib,subprocess,shutil,json;print(json.dumps({"queue":[p.name for p in pathlib.Path("/run/lock/ascend-npu/device-0.queue").iterdir()],"workers":[s for s in subprocess.check_output(["ps","-eo","args"],text=True).splitlines() if s.split() and s.split()[0].endswith("Runner.Worker")],"disk":shutil.disk_usage("/").free}))'
 meta=json.loads(remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
 if meta['queue'] or meta['workers'] or meta['disk']<20_000_000_000:return False,meta
 g=remote(ENV+PY+R+'/guard_private.py status --json',stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
 if g.returncode:return False,{'meta':meta,'status_error':g.stderr,'status':g.stdout}
 state=json.loads(g.stdout);return state.get('state')=='idle' and state.get('reservation',{}).get('locked') is False,{'meta':meta,'guard':state}
def run(family,mode,indices,label):
 checkpoint=P/(label+'.CHECKPOINT.json')
 if checkpoint.exists():return json.loads(checkpoint.read_text())
 tag=label+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');okay,gate=admit();put(P/(tag+'.GATE.json'),gate)
 if not okay:put(P/'ROLLOUT_PROGRESS.json',{'status':'deferred','label':label,'host':H,'gate':gate});return None
 cmd=ENV+PY+R+'/opportunistic_guard.py run --owner root/baseline-ordered100/'+tag+' --wait 10 --timeout 600 --receipt '+R+'/'+tag+'-guard.json -- bash '+R+'/launch.sh '+family+' '+mode+' '+tag+' '+' '.join(map(str,indices))
 put(P/'ROLLOUT_PROGRESS.json',{'status':'device','tag':tag,'indices':indices,'mode':mode,'family':family})
 t=time.perf_counter()
 with (P/(tag+'.log')).open('w') as out:rc=remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
 seconds=time.perf_counter()-t
 receipt=remote('cat '+R+'/'+tag+'-guard.json',stdout=subprocess.PIPE,text=True);g=json.loads(receipt.stdout) if receipt.returncode==0 else {}
 put(P/(tag+'-guard.json'),g)
 event={'label':label,'tag':tag,'indices':indices,'family':family,'mode':mode,'exit_code':rc,'guarded_seconds':seconds,'guard':g}
 with (P/'ROLLOUT_TIMING.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
 if rc==75 and g.get('phase')=='released_without_launch':put(P/'ROLLOUT_PROGRESS.json',{'status':'deferred_without_child','tag':tag});return None
 # Archive failures as well. Missing child/rundir is an explicit metadata-only event.
 exists=remote('test -d '+R+'/runs/'+tag).returncode==0
 if exists:
  archive=tag+'.tar.gz'
  prepare_root={'tileops-scan-denominator-expansion':'prepared-selective-scan-denominator-08f','tileops-scan-bare-denominator':'prepared-selective-scan-bare-08f','torch-scan-grouped-bare':'prepared-selective-scan-bare-08f','torch-gelu3-original-f037':'prepared-torch-gelu3-f037','torch-qk2-original-08f':'prepared-torch-qk2-08f-v2','torch-qk2-native-rms':'prepared-torch-qk2-08f-v2','torch-qk2-native-rms-rope':'prepared-torch-qk2-08f-v2','tileops-qk2-denominator':'prepared-torch-qk2-08f-v2','torch-selective-scan-bare':'prepared-selective-scan-bare-08f'}.get(mode,'prepared-rollout')
  query='import pathlib,json;p=pathlib.Path('+repr(R)+');ids='+repr(indices)+';folder='+repr(prepare_root)+';print(json.dumps([str(pathlib.Path(folder)/str(i)/"prepare.json") for i in ids if (p/folder/str(i)/"prepare.json").is_file()]))'
  available=json.loads(remote('python3 -c '+shlex.quote(query),stdout=subprocess.PIPE,text=True,check=True).stdout)
  event['archived_prepare_metadata']=available;event['missing_prepare_metadata']=[i for i in indices if prepare_root+'/'+str(i)+'/prepare.json' not in available]
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
if __name__=='__main__':
 family,mode,label=sys.argv[1:4];indices=list(map(int,sys.argv[4:]));lock=(P/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 result=run(family,mode,indices,label)
 if result is None:sys.exit(75)
