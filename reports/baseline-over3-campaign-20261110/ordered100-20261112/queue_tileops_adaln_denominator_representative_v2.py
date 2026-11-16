"""One original public TileOps denominator after complete native6; no kernel edits."""
from pathlib import Path
import os,sys,json,datetime,fcntl,subprocess,shlex,time,hashlib
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=P/'TILEOPS_ADALN_DENOMINATOR_REPRESENTATIVE_V2.json';lock=(P/'tileops-adaln-denominator-representative-v2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_prepare_original_denominator','scope':[195],'coverage_increment':0,'not_final_closure_until_source_generated_checker_raw_guard_audit':True,'queued_at':now()}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
put()
try:
 prior=json.loads((P/'OPS_MATH_ADALN3_EXPANSION.json').read_text());assert prior['status']=='AdaLN3_all6_complete_original_candidate_pairs_fullchecker15raw_guard_valid'
 assert not stop()
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);b=P/'rollout_bootstrap.py';text=b.read_text()
  if "'tileops-adaln-denominator-v2':" not in text:
   (P/'rollout_bootstrap.before-tileops-adaln-denominator.py').write_text(text);b.write_text(text.replace('script={',"script={'tileops-adaln-denominator-v2':'worker_tileops_adaln_denominator_v2.py',",1))
  code='from pathlib import Path;import sys;Path('+repr(c.R+'/rollout_bootstrap.py')+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=b.read_bytes(),check=True);assert c.remote('sha256sum '+c.R+'/rollout_bootstrap.py',stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(b.read_bytes()).hexdigest()
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'Unchanged_public_TileOps_AdaLN_denominator_representative195_fullchecker_isolated15','snapshot_read_only':True})
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:
   fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','tileops-adaln-denominator-v2',[195],'rollout-tileops-adaln-denominator-195-v2')
  if e is not None:break
  s['status']='resource_deferred';put();time.sleep(30)
 assert 'batch' in e,'Wholeguard/process/compile fault preserved; diagnose no blindretry'
 results=e['batch']['results'];s.update(status='result_archived_pending_complete_audit',checkpoint='rollout-tileops-adaln-denominator-195-v2.CHECKPOINT.json',result_status=results[0]['status']);put()
 assert len(results)==1 and results[0]['status']=='raw_checked_pending_guard',results
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);s['status']='formal15_raw_checker_guard_valid_pending_generated_denominator_identity_audit'
except BaseException as ex:s.update(status='stopped_requires_diagnosis',error=repr(ex));raise
finally:
 put();c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
