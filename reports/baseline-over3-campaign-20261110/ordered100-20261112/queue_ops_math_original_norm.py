"""Bounded exact original API cases after dynamic privatekernel qualification."""
import pathlib,json,datetime,fcntl,os,sys,subprocess,shlex,hashlib,time
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_ops_math_original_norm import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'ops-math-original-norm-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
state={'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'host':c.H,'queued_at':now(),'status':'source_deployment','scope':[195, 196, 197, 198, 199, 200, 286, 287, 288, 289, 290, 291, 292, 293, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321],'accepted_complete_APIs':0,'not_paired_speedup_or_public_update':True};c.put(P/'OPS_MATH_ORIGINALNORM_EXPANSION.json',state)
try:
 assert not stop()
 while True:
  assert not stop() and json.loads((P.parent/'STATE.json').read_text())['status']=='active'
  f=P/'server2-alignment-20261113/SERVER1_ORIGINAL_MATH_NORM_QUALIFICATION.json'
  if not f.exists():time.sleep(10);continue
  q=json.loads(f.read_text());assert q['status']!='stopped_requires_diagnosis',q
  if 'exit_code' in q:
   assert q['exit_code']==0 and q['guard']['performance_valid'];break
  os.kill(q['pid'],0);time.sleep(10)
 subprocess.run([sys.executable,str(P/'audit_original_math_norm_qualification.py'),'H1'],check=True)
 prior=json.loads((P/'server2-alignment-20261113/ORIGINAL_MATH_NORM_H1_IDENTITY_AUDIT.json').read_text());assert prior['full_API_embedded_kernel_source_build_guard_and_dual_archive_verified']
 proof=json.loads((P/'ORIGINAL_MATH_NORM_CPU_PROOF.json').read_text());assert proof['no_NPU_imported'] and len(proof['checks'])==8
 for name,h in proof['sources'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
 prior['execution_hostname']=json.loads((P/'server2-alignment-20261113'/str(prior['tag']+'.json')).read_text())['runtime']['hostname']
 prior['original_source_sha256']='945f89d805811dcdad2be8938ebbcc48f4965b0e1b5f76b6e4a29f12eb9a57ba'
 prior['formal_source_sha256']={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['ops_math_original_norm_support.py','ops_math_original_norm_layout.py','ops_math_original_norm_cached_fidelity.py','worker_ops_math_original_norm.py','workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/layernorm.py']}
 manifest=P/'server2-alignment-20261113/ORIGINAL_NORM_HOST_QUALIFICATION.json';c.put(manifest,prior)
 dest='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1/ORIGINAL_NORM_HOST_QUALIFICATION.json'
 code='from pathlib import Path;import sys;Path('+repr(dest)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=manifest.read_bytes(),check=True)
 assert c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(manifest.read_bytes()).hexdigest()
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX)
  f=P/'rollout_bootstrap.py';s=f.read_text()
  if "'ops-math-original-norm':" not in s:
   f.with_name('rollout_bootstrap.before-ops-math-original-norm.py').write_text(s);f.write_text(s.replace('script={',"script={'ops-math-original-norm':'worker_ops_math_original_norm.py',",1))
  records={}
  for name in ['rollout_bootstrap.py','worker_ops_math_original_norm.py','ops_math_original_norm_support.py','ops_math_original_norm_layout.py','ops_math_original_norm_cached_fidelity.py','verify_ops_math_original_norm_cached.py']:
   f=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=h
  c.put(P/'OPS_MATH_ORIGINALNORM_DEPLOYMENT.json',{'checked_at':now(),'files':records,'source_and_kernel_unchanged':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'MathOriginalNorm28_pinned_completeLayerNorm_originalAST_isolated15','durable_log':'qualified-coverage-events.jsonl','snapshot_read_only':True})
 for ids in [[195], [196], [197], [198], [199], [200], [286], [287], [288], [289], [290], [291], [292], [293], [308], [309], [310], [311], [312], [313], [314], [315], [316], [317], [318], [319], [320], [321]]:
  assert not stop()
  accepted=set(json.loads((P/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100']);ids=[i for i in ids if i not in accepted]
  holds=json.loads((P/'MATH_ORIGINAL_NORM_REQUIRES_DIAGNOSIS.json').read_text()).get('ordinals',[]) if (P/'MATH_ORIGINAL_NORM_REQUIRES_DIAGNOSIS.json').exists() else [];assert not set(ids)&set(holds),'Failure hold requires diagnosis, no crosshost blindretry'
  if not ids:continue
  code='import pathlib,json;print(json.dumps([i for i in '+repr(ids)+' if not (pathlib.Path('+repr(c.R)+')/"prepared-rollout"/str(i)/"prepare.json").exists()]))'
  missing=json.loads(c.remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if missing:
   with (P/('ops-math-original-norm-prepare-'+str(ids[0])+'.log')).open('w') as out:rc=c.remote('bash '+c.R+'/launch.sh r-template prepare ops-math-original-norm-cpu '+' '.join(map(str,missing)),stdout=out,stderr=subprocess.STDOUT,timeout=600).returncode
   assert rc==0,'CPU-only prepare failure preserved'
  for i in ids:
   with (P/f'ops-math-original-norm-{i}.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_ops_math_original_norm_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
   assert rc==0,'Shared source/payload fault; do not loop failingcases'
  label='rollout-ops-math-original-norm-'+'-'.join(map(str,ids))+'-l1-quiet-v2'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','ops-math-original-norm',ids,label)
   if e is not None:break
   state.update(status='resource_deferred',checked_at=now());c.put(P/'OPS_MATH_ORIGINALNORM_EXPANSION.json',state);time.sleep(30)
  assert 'batch' in e,'Guard/process fault archived; no blindretry'
  bad=[x['ordinal'] for x in e['batch']['results'] if x['status']!='raw_checked_pending_guard']
  if bad:c.put(P/'MATH_ORIGINAL_NORM_REQUIRES_DIAGNOSIS.json',{'ordinals':bad,'checkpoint':label+'.CHECKPOINT.json','no_samecondition_retry':True})
  assert all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']),'Original checker/binding failure preserved; diagnose beforecontinuation'
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);state.update(status='running',last_batch=ids,accepted_complete_APIs=report['accepted_complete_APIs'],checked_at=now());c.put(P/'OPS_MATH_ORIGINALNORM_EXPANSION.json',state)
 state['status']='all28_complete_original_norm_APIs_fullchecker15raw_guard_valid'
except BaseException as exc:state.update(status='stopped_requires_diagnosis',error=repr(exc));raise
finally:
 state['checked_at']=now();c.put(P/'OPS_MATH_ORIGINALNORM_EXPANSION.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'snapshot_read_only':True})
