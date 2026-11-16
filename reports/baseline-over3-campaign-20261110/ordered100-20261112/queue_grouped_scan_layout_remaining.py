"""Exact three original published Torch programs, fresh source/input namespace and full checker."""
from pathlib import Path
import os,sys,json,datetime,fcntl,subprocess,shlex,time,hashlib
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
from summarize_grouped_scan_layout import summarize
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=P/'GROUPED_SCAN_LAYOUT_REMAINING.json';lock=(P/'grouped-scan-layout-remaining-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'source_deployment','scope':[1497,1500,1501,1502],'paired_cases':0,'queued_at':now()}
def put():s['checked_at']=now();c.put(f,s)
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
put()
try:
 assert not stop();proof=json.loads((P/'SELECTIVE_SCAN_ORIGINAL_CPU_DIAGNOSIS.json').read_text());assert proof['no_NPU_imported'] and proof['CPU_published_original_FP32_Torch_program_passes_existing_FP64_oracle_and_original_checker']
 representative=json.loads((P/'GROUPED_SCAN_LAYOUT_PROBE.json').read_text());assert representative['status']=='native_probe_formal_checked_requires_selection_and_expansion_review' and representative['result']['rows'][0]['device_reduction_percent']>=5 and representative['result']['rows'][0]['guard_checker_15raw_valid']
 cpu=json.loads((P/'GROUPED_SCAN_LAYOUT_CPU_PROOF.json').read_text());assert cpu['no_NPU_imported'] and cpu['helper_sha256']==hashlib.sha256((P/'grouped_scan_layout.py').read_bytes()).hexdigest()
 with (P/'controller.lock').open('a') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX);b=P/'rollout_bootstrap.py';text=b.read_text()
  import ast
  mapping=next(n.value for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
  assert ast.literal_eval(mapping.func.value)['torch-scan-grouped-layout-expansion']=='worker_grouped_scan_layout_expansion.py','Actual route must use isolated runtime worker'
  files={}
  for name in ['rollout_bootstrap.py','worker_grouped_scan_layout_expansion.py','selective_scan_cached_fidelity_expansion.py','grouped_scan_layout.py','SELECTIVE_SCAN_ORIGINAL_CPU_DIAGNOSIS.json','GROUPED_SCAN_LAYOUT_CPU_PROOF.json']:
   p=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=p.read_bytes(),check=True);h=hashlib.sha256(p.read_bytes()).hexdigest();assert c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==h;files[name]=h
  c.put(P/'GROUPED_SCAN_LAYOUT_REMAINING_DEPLOYMENT.json',{'checked_at':now(),'files':files,'original_storage_Torch_entry_and_original_committed_checker':True})
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'Original_FP32_scan_B_C_group_views_frozen_CPUFP64_oracle_checker_full15','snapshot_read_only':True})
 for i in s['scope']:
  assert not stop();accepted=set(json.loads((P/'CURRENT_VALID_COVERAGE.json').read_text())['accepted_ordinals_after100'])
  assert i in accepted,'Use already qualified original; do not repeat it'
  if (P/f'rollout-torch-scan-grouped-layout-expansion-{i}-v1.CHECKPOINT.json').exists():continue
  assert not (P/f'GROUPED_SCAN_LAYOUT_{i}_HOLD.json').exists(),'No samecondition numeric/dependency retry'
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:fcntl.flock(slot,fcntl.LOCK_EX);e=c.run('r-template','torch-scan-grouped-layout-expansion',[i],f'rollout-torch-scan-grouped-layout-expansion-{i}-v1')
   if e is not None:break
   s.update(status='resource_deferred',current_ordinal=i);put();time.sleep(30)
  assert 'batch' in e,'Common guard/process fault, preserve and stop'
  r=e['batch']['results'][0]
  if r['status']!='raw_checked_pending_guard':c.put(P/f'GROUPED_SCAN_LAYOUT_{i}_HOLD.json',{'ordinal':i,'checkpoint':f'rollout-torch-scan-grouped-layout-expansion-{i}-v2.CHECKPOINT.json','result':r,'no_samecondition_retry':True});raise RuntimeError(('Original checker/dependency failure needs diagnosis',i,r))
  subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True);summary=summarize();subprocess.run([sys.executable,str(P/'effective_coverage.py')],check=True);s.update(status='running',paired_cases=summary['paired_cases'],current_ordinal=i);put()
 s['status']='remaining4_grouped_layout_candidates_paired'
except BaseException as ex:s.update(status='stopped_requires_diagnosis',error=repr(ex));raise
finally:put();c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
