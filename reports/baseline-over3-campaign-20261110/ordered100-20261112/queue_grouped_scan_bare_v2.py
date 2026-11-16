from pathlib import Path
import os,sys,json,datetime,fcntl,subprocess,shlex,hashlib,time,ast
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_grouped_scan_bare import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=P/'GROUPED_SCAN_BARE_EXPANSION_V2.json';lock=(P/'grouped-scan-bare-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_only_reverse_AST_and_equivalence_preflight','scope':[1499],'queued_at':now()}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
def deploy(name):
 file=P/name;dest=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dest)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=file.read_bytes(),check=True);assert c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(file.read_bytes()).hexdigest()
put()
try:
 assert not stop();assert not (P/'rollout-torch-scan-grouped-bare-1499-v2.CHECKPOINT.json').exists(),'Existing attempt requires evidence review, never repeat'
 # CPU preparation may overlap the previous guarded group, but device dispatch waits for its real termination.
 while True:
  previous=json.loads((P/'CURRENT_PROCESS.json').read_text());pid=previous.get('pid');cmd=Path('/proc')/str(pid)/'cmdline'
  if not pid or not cmd.exists():break
  if 'queue_grouped_scan_layout_remaining.py' not in cmd.read_bytes().decode():raise RuntimeError('Unexpected active controller; no overlapping dispatch')
  s['status']='waiting_previous_group_completion';put();time.sleep(5);assert not stop()
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'CPU_preflight_then_guarded_single_case_original_FP32_grouped_layout_candidate','snapshot_read_only':True})
 for name in ['grouped_scan_layout.py','selective_scan_bare_grouped_device.py','prove_grouped_scan_bare_cpu_v2.py']:deploy(name)
 command='source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh; export TORCH_DEVICE_BACKEND_AUTOLOAD=0; exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+c.R+'/prove_grouped_scan_bare_cpu_v2.py'
 with (P/'grouped-scan-bare-cpu-v2.log').open('w') as out:rc=c.remote(command,stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,'CPU equivalence preflight failure: no native launch';raw=c.remote('cat '+c.R+'/GROUPED_SCAN_BARE_CPU_PROOF_V2.json',stdout=subprocess.PIPE,check=True).stdout;(P/'GROUPED_SCAN_BARE_CPU_PROOF_V2.json').write_bytes(raw);proof=json.loads(raw);assert proof['no_NPU_imported'] and proof['both_outputs_bitexact_vs_original_FP32'] and proof['frozen_FP64_oracle_checker_passed'] and proof['helper_sha256']==hashlib.sha256((P/'grouped_scan_layout.py').read_bytes()).hexdigest()
 (P/'GROUPED_SCAN_BARE_CPU_PROOF.json').write_bytes(raw);deploy('GROUPED_SCAN_BARE_CPU_PROOF.json')
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);tree=ast.parse((P/'rollout_bootstrap.py').read_text());mapping=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets));assert ast.literal_eval(mapping.func.value)['torch-scan-grouped-bare']=='worker_grouped_scan_bare.py'
  for name in ['rollout_bootstrap.py','worker_grouped_scan_bare.py','selective_scan_bare_fidelity.py','selective_scan_bare_device.py','selective_scan_bare_frozen_policy.py','GROUPED_SCAN_BARE_CPU_PROOF_V2.json']:deploy(name)
 s['status']='guarded_native_probe_pending';put()
 while True:
  assert not stop()
  with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);event=c.run('r-template','torch-scan-grouped-bare',[1499],'rollout-torch-scan-grouped-bare-1499-v2')
  if event is not None:break
  s['status']='resource_deferred';put();time.sleep(30)
 assert not (P/'GROUPED_SCAN_BARE_1499_HOLD.json').exists()
 assert 'batch' in event and event['batch']['results'][0]['status']=='raw_checked_pending_guard','Retain failed probe and diagnose; no samecondition retry';subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);result=summarize();s.update(status='bare_grouped_layout_full_frozen_checker_paired',result=result)
except BaseException as ex:s.update(status='stopped_requires_diagnosis',error=repr(ex));raise
finally:put();c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
