"""Bounded exact original API cases after dynamic privatekernel qualification."""
import pathlib,json,datetime,fcntl,os,sys,subprocess,shlex,hashlib,time
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_ops_math_adaln3 import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'ops-math-adaln3-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
state={'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'host':c.H,'queued_at':now(),'status':'source_deployment','scope':[195,196,197,198,199,200],'accepted_complete_pairs':0,'coverage_increment':0,'no_public_or_historical_ratio_update':True};c.put(P/'OPS_MATH_ADALN3_EXPANSION.json',state)
try:
 assert not stop()
 assert json.loads((P.parent/'STATE.json').read_text())['status']=='active'
 while True:
  assert not stop() and json.loads((P.parent/'STATE.json').read_text())['status']=='active'
  f=P/'server2-alignment-20261113/ADALN3_HOST_QUALIFICATION.json'
  if f.exists():break
  time.sleep(5)
 prior=json.loads(f.read_text());assert prior['full_API_original_reference_inputbits_flush_private_kernels_guard_and_dual_archive_verified']
 for name,h in prior['formal_source_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  f=P/'rollout_bootstrap.py';s=f.read_text()
  if "'ops-math-adaln3':" not in s:
   f.with_name('rollout_bootstrap.before-ops-math-adaln3.py').write_text(s);f.write_text(s.replace('script={',"script={'ops-math-adaln3':'worker_ops_math_adaln3.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','worker_ops_math_adaln3.py','ops_math_adaln3_support.py','ops_math_adaln3_layout.py','ops_math_original_norm_cached_fidelity.py','verify_ops_math_original_norm_cached.py','ops_math_add_api.py','ops_math_dequant_api.py','OPS_MATH_ADALN3_CPU_PROOF.json']:
   f=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=h
  c.put(P/'OPS_MATH_ORIGINALNORM_DEPLOYMENT.json',{'checked_at':now(),'files':records,'source_and_kernel_unchanged':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'MathAdaLN3_complete_native_LayerNorm_Mul_Add_pairs_isolated15','durable_log':'qualified-coverage-events.jsonl','snapshot_read_only':True})
 for ids in [[195],[196],[197],[198],[199],[200]]:
  assert not stop()
  accepted={r['ordinal'] for r in json.loads((P/'OPS_MATH_ADALN3_CURRENT.json').read_text())['rows']} if (P/'OPS_MATH_ADALN3_CURRENT.json').exists() else set();ids=[i for i in ids if i not in accepted]
  holds=json.loads((P/'MATH_ADALN3_REQUIRES_DIAGNOSIS.json').read_text()).get('ordinals',[]) if (P/'MATH_ADALN3_REQUIRES_DIAGNOSIS.json').exists() else [];assert not set(ids)&set(holds),'Failure hold requires diagnosis, no crosshost blindretry'
  if not ids:continue
  code='import pathlib,json;print(json.dumps([i for i in '+repr(ids)+' if not (pathlib.Path('+repr(c.R)+')/"prepared-rollout"/str(i)/"prepare.json").exists()]))'
  missing=json.loads(c.remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if missing:
   with (P/('ops-math-adaln3-prepare-'+str(ids[0])+'.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-template prepare ops-math-adaln3-cpu '+' '.join(map(str,missing)),stdout=out,stderr=subprocess.STDOUT,timeout=600).returncode
   assert rc==0,'CPU-only prepare failure preserved'
  for i in ids:
   with (P/f'ops-math-adaln3-{i}.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_ops_math_original_norm_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'Shared source/payload fault; do not loop failingcases'
  label='rollout-ops-math-adaln3-'+'-'.join(map(str,ids))+'-l1-quiet-v2'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','ops-math-adaln3',ids,label)
   if e is not None:break
   state.update(status='resource_deferred',checked_at=now());c.put(P/'OPS_MATH_ADALN3_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in e,'Guard/process fault archived; no blindretry'
  bad=[x['ordinal'] for x in e['batch']['results'] if x['status']!='raw_checked_pending_guard']
  if bad:c.put(P/'MATH_ADALN3_REQUIRES_DIAGNOSIS.json',{'ordinals':bad,'checkpoint':label+'.CHECKPOINT.json','no_samecondition_retry':True})
  assert all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']),'Original checker/binding failure preserved; diagnose beforecontinuation'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();state.update(status='running',last_batch=ids,accepted_complete_pairs=report['paired_cases'],checked_at=now());c.put(P/'OPS_MATH_ADALN3_EXPANSION.json',state)
 state['status']='AdaLN3_all6_complete_original_candidate_pairs_fullchecker15raw_guard_valid'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=now();c.put(P/'OPS_MATH_ADALN3_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
