"""H1 independent exact successful H2 MatMul recipe; no kernel/flag/runtime edits."""
from pathlib import Path
import sys,json,subprocess,shlex,datetime,hashlib,fcntl,os,time
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1';OLD='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-gemm10-v1';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=A/'SERVER1_GEMM_PAD4_GENERATION.json'
def put(v):
 t=f.with_suffix('.new.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(f)
lock=(D/'server1-gemm-pad4-generation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'entry':Path(__file__).name,'pid':os.getpid(),'host':c.H,'status':'CPU_existingcanonicalreference_preflight','checked_at':now(),'not_performance_or_coverage':True};put(s)
try:
 reference=json.loads((A/'SERVER2_GEMM10_GENERATION.json').read_text());assert reference['exit_code']==0 and reference['guard']['performance_valid'];prior=json.loads((A/'SERVER2_GEMM10_FP32_COMPILE_V2.json').read_text());assert prior['exit_code']==0 and prior['guard']['performance_valid']
 code='from pathlib import Path;import hashlib,json;n=Path('+repr(OLD)+');m=json.loads((n/"SOURCE_MANIFEST.json").read_text());assert hashlib.sha256((n/"SOURCE_MANIFEST.json").read_bytes()).hexdigest()=="195fe88b744c6a2336720e3a028d0dfdf8c0648047dfa895e12219a6630006f1";assert all(hashlib.sha256((n/"source"/k).read_bytes()).hexdigest()==v["sha256"] for k,v in m["files"].items());print(json.dumps({"verified_source_files":len(m["files"]),"build_recipe_sha256":hashlib.sha256((n/"build-device-entry.sh").read_bytes()).hexdigest(),"source_pin":m["pin"]}))'
 r=c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True);proof=json.loads(r.stdout);assert proof['verified_source_files']==25610 and proof['source_pin']=='84bf2da40119e3b8114a6d2a9e7369190a0470d1';assert proof['build_recipe_sha256']==hashlib.sha256((A/'native-gemm10-generate-entry.sh').read_bytes()).hexdigest();s['same_successful_H2_recipe_and_source']=proof;put(s)
 while True:
  assert not (D/'STOP').exists() and not (D.parent/'STOP').exists()
  with (D/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);okay,gate=c.admit()
   if okay:
    tag='server1-gemm-pad4-generation-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');receipt=N+'/'+tag+'-guard.json';s.update(status='guarded_samecanonical_recipe_generation',tag=tag,checked_at=now());put(s);(A/(tag+'.GATE.json')).write_text(json.dumps(gate,indent=2));command='source '+ROOT+'/environment.sh; exec '+ROOT+'/job-python-v1/bin/python '+c.R+'/opportunistic_guard.py run --owner root/baseline-native-build/'+tag+' --wait 10 --timeout 600 --receipt '+receipt+' -- bash '+OLD+'/build-device-entry.sh'
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
