"""Single390 complete existing LayerNorm affine; manifest-driven provider routing."""
from pathlib import Path
import os,sys,json,datetime,subprocess,shlex,fcntl,hashlib,time
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';sys.path.insert(0,str(P));import rollout_controller as c
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-adaln-decode-ln-v1';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';JOB=ROOT+'/job-python-v1/bin/python';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();statefile=A/'SERVER1_ADALN_DECODE_LN_PROBE_V2.json';lock=(P/'decode-layernorm-manifest-route.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'queued_at':now(),'status':'CPU_manifest_route_preflight','scope':[390],'not_formal_measurement_or_coverage':True}
def put():s['checked_at']=now();c.put(statefile,s)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
put()
try:
 assert not stop();proof=json.loads((P/'OPS_NN_ADALN_DECODE_LN_CPU_PROOF.json').read_text());assert proof['no_NPU_imported'] and len(proof['rows'])==1
 code='from pathlib import Path;import json,hashlib;n=Path('+repr(N)+');m=json.loads((n/"LAYERNORM_BASE_PACKAGE.json").read_text());p=Path(m["staging_root"]);assert (p/"op_impl/ai_core/tbe/kernel/config/ascend950/layer_norm_v4_apt.json").is_file();f=p/"op_api/lib/libcust_opapi.so";assert hashlib.sha256(f.read_bytes()).hexdigest()==m["libraries"]["op_api/lib/libcust_opapi.so"]["sha256"];assert (n/"probe_adaln_decode_layernorm.py").is_file();print(json.dumps({"vendor":str(p),"manifest_sha256":hashlib.sha256((n/"LAYERNORM_BASE_PACKAGE.json").read_bytes()).hexdigest(),"probe_sha256":hashlib.sha256((n/"probe_adaln_decode_layernorm.py").read_bytes()).hexdigest()}))'
 pre=json.loads(c.remote('python3 -I -S -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout);assert pre['probe_sha256']==hashlib.sha256((A/'probe_adaln_decode_layernorm.py').read_bytes()).hexdigest();c.put(A/'ADALN_DECODE_LN_MANIFEST_ROUTE_PREFLIGHT.json',pre)
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'single390_full_LayerNorm_affine_manifest_verified_private_provider_probe_not15raw','snapshot_read_only':True})
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);okay,gate=c.admit()
   if okay:
    tag='server1-adaln-decode-ln-v2-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');owner='root/baseline-native-probe/'+tag;result=tag+'.json';receipt=N+'/'+tag+'-guard.json';v=pre['vendor'];dirs=[v+'/op_api/lib',v+'/op_proto/lib/linux/aarch64',v+'/op_impl/ai_core/tbe/op_tiling',v+'/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64'];env='env ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_CUSTOM_OPP_PATH='+shlex.quote(v)+' LD_LIBRARY_PATH='+':'.join(dirs)+':"$LD_LIBRARY_PATH" ';cmd='source '+ROOT+'/native/ops-math-layernorm-original-v1/runtime-env-formal.sh; exec '+JOB+' '+c.R+'/opportunistic_guard.py run --owner '+owner+' --wait 10 --timeout 600 --receipt '+receipt+' -- '+env+JOB+' '+N+'/probe_adaln_decode_layernorm.py '+owner+' '+result;c.put(A/(tag+'.GATE.json'),gate);s.update(status='guarded_complete_API_probe',tag=tag);put()
    with (A/(tag+'.log')).open('w') as out:r=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT)
    raw=c.remote('cat '+receipt,stdout=subprocess.PIPE,check=True).stdout;(A/(tag+'-guard.json')).write_bytes(raw);g=json.loads(raw)
   else:s.update(status='resource_deferred',gate=gate);put()
  if not okay:time.sleep(30);continue
  if r.returncode==75 and g.get('phase')=='released_without_launch':continue
  raw=c.remote('cat '+N+'/'+result,stdout=subprocess.PIPE,check=True).stdout;(A/result).write_bytes(raw);data=json.loads(raw);s.update(status='functional_passed_private_attribution_pending' if r.returncode==0 else 'numeric_or_dependency_failure_requires_diagnosis_no_retry',exit_code=r.returncode,guard=g,result=result,rows=data.get('rows',[]))
  for f in [A/(tag+'.log'),A/(tag+'-guard.json'),A/result]:
   dest=N+'/'+f.name;code='from pathlib import Path;import sys;p=Path('+repr(dest)+');raw=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw)';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();s.setdefault('dual_archive',[]).append({'path':dest,'sha256':h})
  break
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:put();c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
