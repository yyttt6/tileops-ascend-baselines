import pathlib,sys,json,fcntl,datetime,time,tarfile
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import controller as c
family,label=sys.argv[1:3];ids=list(map(int,sys.argv[3:]));lock=(P/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
okay,gate=c.admit()
if not okay:c.put(P/'CHECK_PROGRESS.json',{'status':'deferred','label':label,'gate':gate});sys.exit(75)
tag=label+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');c.put(P/'CHECK_PROGRESS.json',{'status':'canonical_correctness_only','tag':tag,'ids':ids})
cmd=c.ENV+c.PY+c.R+'/opportunistic_guard.py run --owner root/baseline-canonical-check/'+tag+' --wait 10 --timeout 600 --receipt '+c.R+'/'+tag+'-guard.json -- bash '+c.R+'/launch.sh '+family+' '+tag+' '+' '.join(map(str,ids));start=time.perf_counter()
with (P/(tag+'.log')).open('w') as f:rc=c.remote(cmd,stdout=f,stderr=__import__('subprocess').STDOUT).returncode
s=c.remote('cat '+c.R+'/'+tag+'-guard.json',stdout=__import__('subprocess').PIPE,text=True);g=json.loads(s.stdout) if s.returncode==0 else {};c.put(P/(tag+'-guard.json'),g)
e={'tag':tag,'ids':ids,'family':family,'exit_code':rc,'guard':g,'wall_seconds':time.perf_counter()-start,'performance_windows':0}
if c.remote('test -d '+c.R+'/runs/'+tag).returncode==0:
 a=tag+'.tar.gz';c.remote('cd '+c.R+' && tar -czf '+a+' runs/'+tag+' '+tag+'-guard.json',check=True)
 with (P/a).open('wb') as f:c.remote('cat '+c.R+'/'+a,stdout=f,check=True)
 digest=c.remote('sha256sum '+c.R+'/'+a,stdout=__import__('subprocess').PIPE,text=True,check=True).stdout.split()[0];assert digest==c.sha(P/a)
 with tarfile.open(P/a) as t:t.extractall(P/'remote',filter='data')
 e['archive_sha256']=digest
 b=P/'remote/runs'/tag/'BATCH.json'
 if b.exists():e['batch']=json.loads(b.read_text())
if rc==75 and g.get('phase')=='released_without_launch':
 c.put(P/(tag+'.DEFERRED.json'),e);sys.exit(75)
c.put(P/(label+'.CANONICAL.json'),e)
assert rc==0 and g.get('performance_valid') and g.get('workload_exit_code')==0,e
print('CHECKED',label,[(r['ordinal'],r['status']) for r in e.get('batch',{}).get('results',[])],flush=True)
