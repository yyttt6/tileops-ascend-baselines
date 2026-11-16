"""Wait for our active group, then add frozen checker evidence without timing reruns."""
from pathlib import Path
import os,sys,json,subprocess,shlex,hashlib,datetime,fcntl,time,tarfile
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=P/'QK2_FROZEN_CHECKER_PROCESS_V2.json';lock=(P/'qk2-frozen-checker-v2-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'await_own_scan_group_release','queued_at':now(),'scope':[1356,1358],'correctness_only_no_performance_rerun':True}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
put()
try:
 assert not stop()
 while True:
  prior=json.loads((P/'SELECTIVE_SCAN_ORIGINAL_REMAINING.json').read_text());pid=prior['pid'];proc=Path('/proc')/str(pid)
  if not proc.exists() or proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
  assert 'queue_selective_scan_original_remaining.py' in proc.joinpath('cmdline').read_bytes().decode(),'Never wait on a reused unrelated PID'
  assert not stop();time.sleep(5)
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'Frozen_checker_supplement_correctness_only_QK2_all4_roles_no_profiler_or_timing','snapshot_read_only':True})
 file=P/'qk2_frozen_checker_supplement_v2.py';dest=c.R+'/'+file.name;code='from pathlib import Path;import sys;Path('+repr(dest)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=file.read_bytes(),check=True);assert c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(file.read_bytes()).hexdigest()
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);okay,gate=c.admit()
   if okay:
    tag='qk2-frozen-checker-v2-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');receipt=c.R+'/'+tag+'-guard.json';shell='source /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1/runtime-env-formal.sh; export TORCH_DEVICE_BACKEND_AUTOLOAD=0; exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+dest+' '+tag;command=c.ENV+c.PY+c.R+'/opportunistic_guard.py run --owner root/baseline-frozen-checker/'+tag+' --wait 10 --timeout 240 --receipt '+receipt+' -- bash -c '+shlex.quote(shell);s.update(status='guarded_correctness_only',tag=tag,gate=gate);put()
    with (P/(tag+'.log')).open('w') as out:rc=c.remote(command,stdout=out,stderr=subprocess.STDOUT).returncode
    guard=json.loads(c.remote('cat '+receipt,stdout=subprocess.PIPE,text=True,check=True).stdout);c.put(P/(tag+'-guard.json'),guard)
    if rc==75 and guard['phase']=='released_without_launch':okay=False
    else:break
  s.update(status='resource_deferred',gate=gate);put();time.sleep(30)
 assert rc==0 and guard['phase']=='finished' and guard['workload_exit_code']==0 and guard['performance_valid'],'Frozen checker supplement must actually pass; preserve failed evidence and keep gate held'
 raw=c.remote('cat '+c.R+'/runs/'+tag+'/QK2_FROZEN_CHECKER_SUPPLEMENT.json',stdout=subprocess.PIPE,check=True).stdout;proof=json.loads(raw);assert proof['status']=='passed' and len(proof['rows'])==8
 archive=tag+'.tar.gz';c.remote('cd '+c.R+' && tar -czf '+archive+' runs/'+tag+' '+tag+'-guard.json qk2_frozen_checker_supplement_v2.py',check=True)
 with (P/archive).open('wb') as out:c.remote('cat '+c.R+'/'+archive,stdout=out,check=True)
 digest=hashlib.sha256((P/archive).read_bytes()).hexdigest();assert c.remote('sha256sum '+c.R+'/'+archive,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==digest
 proof.update(guard_receipt=tag+'-guard.json',guard_receipt_sha256=hashlib.sha256((P/(tag+'-guard.json')).read_bytes()).hexdigest(),archive=archive,archive_sha256=digest,archive_both_local_and_remote_SHA_verified=True)
 c.put(P/'QK2_FROZEN_CHECKER_SUPPLEMENT.json',proof);subprocess.run([sys.executable,str(P/'summarize_torch_qk2_final.py')],check=True);subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);s.update(status='all8_frozen_checker_supplements_passed_original15raw_preserved',tag=tag)
except BaseException as ex:s.update(status='stopped_requires_diagnosis',error=repr(ex));raise
finally:put();c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
