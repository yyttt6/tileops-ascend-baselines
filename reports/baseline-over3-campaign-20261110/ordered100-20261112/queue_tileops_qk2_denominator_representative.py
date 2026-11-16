"""Bounded unchanged public QK denominator; require generated identity before expansion."""
from pathlib import Path
import os,sys,json,datetime,fcntl,subprocess,shlex,hashlib,time,ast
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=P/'TILEOPS_QK2_DENOMINATOR_REPRESENTATIVE.json';lock=(P/'tileops-qk2-denominator-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'source_deployment','scope':[1356],'qualified_denominators':0,'queued_at':now()}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
put()
try:
 assert not stop();pairs=json.loads((P/'TORCH_QK2_NATIVE_RMS_CURRENT.json').read_text());assert pairs['paired_cases']==2
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);b=P/'rollout_bootstrap.py';tree=ast.parse(b.read_text());mapping=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets));assert ast.literal_eval(mapping.func.value)['tileops-qk2-denominator']=='worker_tileops_qk2_denominator.py'
  files={}
  for name in ['rollout_bootstrap.py','worker_tileops_qk2_denominator.py','torch_qk2_original_fidelity.py','tileops_qk2_denominator_support.py']:
   file=P/name;dest=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dest)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=file.read_bytes(),check=True);digest=hashlib.sha256(file.read_bytes()).hexdigest();assert c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==digest;files[name]=digest
  c.put(P/'TILEOPS_QK2_DENOMINATOR_DEPLOYMENT.json',{'checked_at':now(),'files':files,'original_public_Op_and_binder':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'Unchanged_original08f_public_TileOps_QK_denominator_representative','snapshot_read_only':True})
 for i in s['scope']:
  assert not stop();assert not (P/f'TILEOPS_QK2_DENOMINATOR_{i}_HOLD.json').exists(),'No samecondition retry'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','tileops-qk2-denominator',[i],f'rollout-tileops-qk2-denominator-{i}-v1')
   if e is not None:break
   s.update(status='resource_deferred',current_ordinal=i);put();time.sleep(30)
  assert 'batch' in e,'Common guard/process failure must be diagnosed'
  result=e['batch']['results'][0]
  if result['status']!='raw_checked_pending_guard':c.put(P/f'TILEOPS_QK2_DENOMINATOR_{i}_HOLD.json',{'ordinal':i,'checkpoint':e['label']+'.CHECKPOINT.json','result':result,'no_samecondition_retry':True});raise RuntimeError(('Original public TileOps denominator failed',i,result))
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
  subprocess.run([sys.executable,str(P/'audit_tileops_qk2_denominator_identity.py'),str(i)],check=True)
  subprocess.run([sys.executable,str(P/'summarize_torch_qk2_final.py')],check=True)
  s.update(status='running',qualified_denominators=s['qualified_denominators']+1,current_ordinal=i);put()
 s['status']='representative_raw_checker_guard_generated_identity_qualified'
except BaseException as ex:s.update(status='stopped_requires_diagnosis',error=repr(ex));raise
finally:put();c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
