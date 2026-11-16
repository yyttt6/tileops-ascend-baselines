"""H2 canonical reference first; whole-task gates do not block independent H1 campaign."""
from pathlib import Path
import os,sys,json,time,datetime,fcntl,subprocess,shlex,hashlib,tarfile
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_mathl1_h2 as c
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';f=A/'ORIGINAL_MATH_NORM_H2_QUALIFICATION.json';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();lock=(D/'original-math-norm-h2-qualification.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'role':'support','status':'CPU_wait_H2_whole_task_release','queued_at':now(),'zero_formal_coverage':True,'H1_independent':True}
def put():
 s['checked_at']=now();c.put(f,s)
def stopped():return (D/'STOP').exists() or (D.parent/'STOP').exists() or json.loads((D.parent/'STATE.json').read_text())['status']!='active'
put()
try:
 build=json.loads((A/'ORIGINAL_MATH_LAYERNORM_H2_CPU_BUILD.json').read_text());assert build['status']=='CPU_build_complete_runtime_qualification_required' and build['no_NPU_modules_loaded'];align=json.loads((A/'ORIGINAL_MATH_LAYERNORM_CPU_BUILD_ALIGNMENT.json').read_text());assert align['original_pin_source_gitblob_compiler_CXX_Torch_ABI_and_normalized_commands_equal'];cpu=json.loads((D/'ORIGINAL_MATH_NORM_CPU_PROOF.json').read_text());assert len(cpu['checks'])==8 and cpu['no_NPU_imported']
 while True:
  assert not stopped();okay,gate=c.admit();s.update(status='fresh_H2_whole_task_release_check',gate=gate);put()
  if not okay:s['status']='H2_task_resource_deferred_H1_continues';put();time.sleep(60);continue
  tag='original-math-norm-h2-qualification-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');owner='root/baseline-native-qualification/server2/'+tag;receipt=N+'/'+tag+'-guard.json';result=tag+'.json';c.put(A/(tag+'.GATE.json'),gate);s.update(status='guarded_original_kernel_load_run',tag=tag);put();command='source '+ROOT+'/environment.sh; exec '+c.JOB+' '+c.GUARD+' run --owner '+owner+' --wait 10 --timeout 180 --receipt '+receipt+' -- bash '+N+'/probe-entry.sh '+owner+' '+result
  with (A/(tag+'.log')).open('w') as out:rc=c.remote(command,stdout=out,stderr=subprocess.STDOUT).returncode
  raw=c.remote('cat '+receipt,stdout=subprocess.PIPE,check=True).stdout;g=json.loads(raw);(A/(tag+'-guard.json')).write_bytes(raw)
  if rc==75 and g.get('phase')=='released_without_launch':s.update(status='H2_not_launched_resource_backoff',tag=tag);put();time.sleep(60);continue
  for suffix in ['.json','.json.maps.txt','.json.pt']:
   raw=c.remote('cat '+N+'/'+tag+suffix,stdout=subprocess.PIPE,check=True).stdout;p=A/(tag+suffix);p.write_bytes(raw);assert c.remote('sha256sum '+N+'/'+p.name,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(raw).hexdigest()
  data=json.loads((A/result).read_text());s.update(status='functional_passed_embedded_kernel_identity_audit_required' if rc==0 and g.get('performance_valid') and data.get('full_output_and_inputs_verified') else 'failed_requires_diagnosis_no_retry',exit_code=rc,guard=g,result=result);put()
  assert rc==0 and g['phase']=='finished' and g['performance_valid'] is True and data['full_output_and_inputs_verified'],'Failure preserved, no numerical or ABI blindretry'
  assert data['loaded_original_binary_sha256']==build['binary_sha256'] and data['loaded_original_private_binary_present'] and len(data['checks'])==4 and data['original_1GiB_flush_verified'];break
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));put();raise
