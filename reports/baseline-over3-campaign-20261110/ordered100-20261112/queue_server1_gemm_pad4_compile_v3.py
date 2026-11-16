"""Freshguard bounded compiler checkpoints; preserve8completed commands and failedbatch."""
from pathlib import Path
import sys,subprocess,json,shlex,datetime,fcntl,os,time,hashlib
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();file=A/'SERVER1_GEMM_PAD4_COMPILE_V3.json'
def put():
 t=file.with_suffix('.next.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(file)
def stop():return (D/'STOP').exists() or (D.parent/'STOP').exists()
lock=(D/'server1-gemm-pad4-compile-v3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'entry':Path(__file__).name,'pid':os.getpid(),'host':c.H,'status':'source_deploy_bounded2','checked_at':now(),'initial_complete_commands_preserved':8,'max_original_commands_per_guard':2,'batches':[],'not_performance_or_formal_coverage':True};put()
try:
 prior=json.loads((A/'SERVER1_GEMM_PAD4_COMPILE.json').read_text());assert prior['exit_code']==124 and prior['guard']['workload_exit_code']!=0
 for src,dst in [(A/'compile_gemm_pad4_batches_v3.py',N+'/compile_gemm_pad4_batches_v3.py'),(A/'gemm-pad4-compile-batch-entry-v3.sh',N+'/compile-batch-entry-v3.sh')]:
  raw=src.read_bytes();code='from pathlib import Path;import sys,hashlib;p=Path('+repr(dst)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b);print(hashlib.sha256(b).hexdigest())';r=c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=raw,stdout=subprocess.PIPE,check=True);assert r.stdout.decode().strip()==hashlib.sha256(raw).hexdigest()
 for args in [[10,11],[12,13],[14,15],[16,17],[8,9],['finalize']]:
  assert not stop()
  code='from pathlib import Path;import json;n=Path('+repr(N)+');p=json.loads((n/"PAD4_COMPILE_SELECTION_H1.json").read_text());l=json.loads((n/"PAD4_COMPILE_RESUME_LEDGER.json").read_text());print(json.dumps([i for i in '+repr(args)+' if p["selected_whole_commands"][i]["command_sha256"] not in l["compiled"]]))' if args!=['finalize'] else None
  if code:
   args=json.loads(c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
   if not args:continue
  while True:
   assert not stop()
   with (D/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);okay,gate=c.admit()
    if okay:
     tag='server1-gemm-pad4-part-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');receipt=N+'/'+tag+'-guard.json';s.update(status='guarded_original_commands',current=args,tag=tag,checked_at=now());put();(A/(tag+'.GATE.json')).write_text(json.dumps(gate,indent=2));cmd='source '+ROOT+'/environment.sh; exec '+ROOT+'/job-python-v1/bin/python '+c.R+'/opportunistic_guard.py run --owner root/baseline-native-build/'+tag+' --wait 10 --timeout 600 --receipt '+receipt+' -- bash '+N+'/compile-batch-entry-v3.sh '+shlex.join([str(i) for i in args])
     with (A/(tag+'.log')).open('w') as out:r=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT)
     raw=c.remote('cat '+receipt,stdout=subprocess.PIPE,check=True).stdout;(A/(tag+'-guard.json')).write_bytes(raw);g=json.loads(raw)
   if not okay:s.update(status='H1_resource_deferred',checked_at=now());put();time.sleep(30);continue
   if r.returncode==75 and g.get('phase')=='released_without_launch':time.sleep(30);continue
   for f in [A/(tag+'.log'),A/(tag+'-guard.json')]:
    raw=f.read_bytes();dest=N+'/'+f.name;code='from pathlib import Path;import sys,hashlib;p=Path('+repr(dest)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b);print(hashlib.sha256(b).hexdigest())';rr=c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),input=raw,stdout=subprocess.PIPE,check=True);assert rr.stdout.decode().strip()==hashlib.sha256(raw).hexdigest()
   s['batches'].append({'args':args,'tag':tag,'exit_code':r.returncode,'guard':g});s.update(checked_at=now());put();assert r.returncode==0 and g.get('phase')=='finished' and g.get('workload_exit_code')==0 and g.get('performance_valid'),('Boundedpartfailed; no blindretry',args,r.returncode)
   ledger=c.remote('cat '+N+'/PAD4_COMPILE_RESUME_LEDGER.json',stdout=subprocess.PIPE,check=True).stdout;(A/'PAD4_COMPILE_RESUME_LEDGER.current.json').write_bytes(ledger);s['compiled_commands']=len(json.loads(ledger)['compiled']);put();break
 s.update(status='all18_compiler_commands_metadata_complete',exit_code=0,guard=g,checked_at=now());put()
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e),checked_at=now());raise
finally:put()
