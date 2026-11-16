"""H1 independent exact successful H2 MatMul recipe; no kernel/flag/runtime edits."""
from pathlib import Path
import sys,json,subprocess,shlex,datetime,hashlib,fcntl,os,time
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-pool-mean-v1';OLD='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=A/'SERVER1_MATH_POOL_MEAN_GENERATION.json'
def put(v):
 t=f.with_suffix('.new.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(f)
lock=(D/'server1-math-pool-mean-generation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'entry':Path(__file__).name,'pid':os.getpid(),'host':c.H,'status':'CPU_existingcanonicalreference_preflight','checked_at':now(),'not_performance_or_coverage':True};put(s)
try:
 reference=json.loads((A/'SERVER2_OPSMATH_LEAF4_HOST.json').read_text());assert reference['exit_code']==0 and reference['guard']['performance_valid']
 proof=json.loads((A/'MATH_POOL_MEAN_H1_SOURCE.json').read_text());assert proof['verified_source_files']==14279 and proof['original_files_unchanged'];s['same_successful_H2_math_build_entry_source']=proof;put(s)
 while True:
  assert not (D/'STOP').exists() and not (D.parent/'STOP').exists()
  with (D/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);okay,gate=c.admit()
   if okay:
    tag='server1-math-pool-mean-generation-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');receipt=N+'/'+tag+'-guard.json';s.update(status='guarded_samecanonical_recipe_generation',tag=tag,checked_at=now());put(s);(A/(tag+'.GATE.json')).write_text(json.dumps(gate,indent=2));command='source '+ROOT+'/environment.sh; exec '+ROOT+'/job-python-v1/bin/python '+c.R+'/opportunistic_guard.py run --owner root/baseline-native-build/'+tag+' --wait 10 --timeout 600 --receipt '+receipt+' -- bash '+N+'/generation-entry.sh'
    with (A/(tag+'.log')).open('w') as out:r=c.remote(command,stdout=out,stderr=subprocess.STDOUT)
    raw=c.remote('cat '+receipt,stdout=subprocess.PIPE,check=True).stdout;(A/(tag+'-guard.json')).write_bytes(raw);g=json.loads(raw)
   else:s.update(status='H1_resource_deferred_independent_of_H2',checked_at=now());put(s)
  if not okay:time.sleep(30);continue
  if r.returncode==75 and g.get('phase')=='released_without_launch':time.sleep(30);continue
  s.update(exit_code=r.returncode,guard=g,status='generated_pending_originalcommands_review' if r.returncode==0 and g.get('performance_valid') else 'failed_requires_diagnosis_no_retry',checked_at=now())
  for f2 in [A/(tag+'.log'),A/(tag+'-guard.json')]:
   raw=f2.read_bytes();code='from pathlib import Path;import sys,hashlib;p=Path('+repr(N+'/'+f2.name)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b);print(hashlib.sha256(b).hexdigest())';r2=c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=raw,stdout=subprocess.PIPE,check=True);assert r2.stdout.decode().strip()==hashlib.sha256(raw).hexdigest()
  put(s);assert s['exit_code']==0 and g.get('performance_valid');break
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e),checked_at=now());raise
finally:put(s)
