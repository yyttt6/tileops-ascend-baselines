"""Archive interrupted stdout workload before resuming; no valid checkpoint repeated."""
import pathlib,sys,json,subprocess,shlex,tarfile,hashlib,datetime,os,fcntl
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
lock=(P/'foreground-recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.loads((P/'CURRENT_PROCESS.json').read_text())['pid'];assert not pathlib.Path('/proc/'+str(old)).exists()
tag='rollout-1948-1950-cache-evidence2-045616292644';label='rollout-1948-1950-cache-evidence2';assert not (P/(label+'.CHECKPOINT.json')).exists()
r=c.remote('cat '+c.R+'/'+tag+'-guard.json',stdout=subprocess.PIPE,text=True,check=True);g=json.loads(r.stdout);assert g['phase']=='finished' and g['exit_code']==120 and not g['performance_valid']
r=c.remote('cat '+c.R+'/runs/'+tag+'/1948/result.json',stdout=subprocess.PIPE,text=True,check=True);failure=json.loads(r.stdout);assert 'BrokenPipeError' in failure['error'] and "print('VERIFIED'" in failure['traceback']
archive=tag+'.interrupted.tar.gz';members=['runs/'+tag,tag+'-guard.json']+['prepared-rollout/'+str(i)+'/prepare.json' for i in [1948,1949,1950]]
c.remote('cd '+c.R+' && tar -I '+shlex.quote('gzip -1')+' -cf '+archive+' '+' '.join(members),check=True)
with (P/archive).open('wb') as out:c.remote('cat '+c.R+'/'+archive,stdout=out,check=True)
expected=c.remote('sha256sum '+c.R+'/'+archive,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert hashlib.sha256((P/archive).read_bytes()).hexdigest()==expected
with tarfile.open(P/archive) as t:t.extractall(P/'remote',filter='data')
e={'label':label,'tag':tag,'family':'r-tla-original','mode':'auto','indices':[1948,1949,1950],'exit_code':120,'guard':g,'archive_sha256':expected,'no_performance_accepted':True,'failure':failure,'reason':'BrokenPipe while printing VERIFIED; current controller/worker absent; not numerical failure, no valid batch/checkpoint'}
(P/(label+'.FAILED.json')).write_text(json.dumps(e,indent=2));(P/'LOCAL_FOREGROUND_INTERRUPTION.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'previous_controller_pid':old,'remote_current_workers_absent':True,'event':e,'cause_of_local_process_exit_unknown':True,'recovery':'foreground processes with durable stdout/stderr files, no daemon or processgroup escape; fullguard still mandatory','existing_valid_checkpoints_not_repeated':True},indent=2))
(P/'CURRENT_PROCESS.json').write_text(json.dumps({'pid':os.getpid(),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entry':'continue_ordered.py','foreground_tool_session':'see FOREGROUND_RECOVERY_TOOL.json','own_migration_disabled':True},indent=2))
os.execv(sys.executable,[sys.executable,'-u',str(P/'continue_ordered.py')])
