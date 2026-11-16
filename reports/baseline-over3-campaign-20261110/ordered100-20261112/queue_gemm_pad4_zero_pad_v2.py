from pathlib import Path
import sys,json,hashlib,subprocess,shlex,datetime,fcntl,os,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_gemm_pad4_zero_pad import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();lock=(P/'gemm-pad4-zero-pad-v2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);file=P/'OPS_NN_GEMM_PAD4_ZERO_PAD_EXPANSION_V2.json';s={'entry':Path(__file__).name,'pid':os.getpid(),'status':'CPU_control_identity','scope':[497],'queued_at':now(),'coverage_increment':0,'original_control_reused_no_rerun':True}
def put():c.put(file,s)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
put()
try:
 assert not stop();base=P/'rollout-ops-nn-gemm-pad4-497-l1-v1.CHECKPOINT.json';control=json.loads(base.read_text());assert control['guard']['performance_valid'] and control['batch']['results'][0]['status']=='raw_checked_pending_guard';proof=json.loads((P/'OPS_NN_GEMM_PAD4_ZERO_PAD_CPU_PROOF.json').read_text());assert proof['full_reverse_AST_identical'] and proof['CPU_symbolic_shape_dtype_value_checks']==8 and proof['nonzero_padding_rejected']==4
 libs=json.loads((P/'remote/runs'/control['tag']/'LOADED_LIBRARIES.json').read_text());code='from pathlib import Path;import hashlib,json\ndef sha(p):\n h=hashlib.sha256()\n with Path(p).open("rb") as f:\n  for b in iter(lambda:f.read(8388608),b""):h.update(b)\n return h.hexdigest()\nm=json.loads('+repr(json.dumps(libs))+');bad={p for p,h in m.items() if not Path(p).is_file() or sha(p)!=h};assert not bad,bad;print(len(m))';r=c.remote('/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True);s['current_control_library_bytes_reverified']=int(r.stdout);put()
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);p=P/'rollout_bootstrap.py';old=p.read_text()
  if "'gemm-pad4-zero-pad-v2':" not in old:p.with_name('rollout_bootstrap.before-gemm-pad4-zero-pad.py').write_text(old);p.write_text(old.replace('script={',"script={'gemm-pad4-zero-pad-v2':'worker_ops_nn_gemm_pad4_zero_pad_v2.py',",1))
  for name in ['rollout_bootstrap.py','worker_ops_nn_gemm_pad4_zero_pad_v2.py','ops_nn_gemm_pad4_zero_pad_support.py','OPS_NN_GEMM_PAD4_ZERO_PAD_CPU_PROOF.json']:
   f=P/name;c.remote('cat > '+c.R+'/'+name,input=f.read_bytes(),check=True);assert c.remote('sha256sum '+c.R+'/'+name,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(f.read_bytes()).hexdigest()
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'phase':'oneF16zero_pad_candidate_samecompleteAPI_originalchecker15','checked_at':now(),'durable_log':'gemm-pad4-zero-pad-v2-queued.log'});s.update(status='candidate_originalisolated15',checked_at=now());put()
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','gemm-pad4-zero-pad-v2',[497],'rollout-gemm-pad4-zero-pad-497-v2')
  if e is not None:break
  s.update(status='resource_deferred',checked_at=now());put();time.sleep(30)
 assert 'batch' in e and all(r['status']=='raw_checked_pending_guard' for r in e['batch']['results']),'Candidate failure preserved; no blindretry';subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);report=summarize();s.update(status='paired_originalchecker15_completed',paired_cases=1,selection=report['rows'][0]['selection'],checked_at=now());put()
except BaseException as ex:s.update(status='stopped_requires_diagnosis',error=repr(ex),checked_at=now());raise
finally:put();c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'outcome':s['status'],'checked_at':now()})
