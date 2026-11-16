"""Reuse qualifiedsamepin H2 framework/profile for independentlyselected originalH1ops."""
from pathlib import Path
import json,datetime,fcntl,os,sys,subprocess,shlex,hashlib,time
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';sys.path.insert(0,str(P));import rollout_controller as c
ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';N=ROOT+'/native/ops-nn-unary-math-v1';JOB=ROOT+'/job-python-v1/bin/python'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'native-math-build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prior_reuse=json.loads((P/'OPS_NN_MATH_DEVICE_ALIAS_RECOVERY.json').read_text())
s={'pid':os.getpid(),'entry':Path(__file__).name,'host':c.H,'queued_at':now(),'status':'wait_own_PTO_terminal','selected_ops':['foreach_exp','foreach_reciprocal'],'no_baseline_coverage_from_build':True,'H2nightly_doesnotblockH1':True,'stages':[prior_reuse['host_stage']]};c.put(P/'OPS_NN_MATH_NATIVE_BUILD.json',s);owns_current=False
try:
 assert not stop();plan=json.loads((P/'OPS_NN_UNARY_MATH_PLAN.json').read_text());assert plan['source_files_bothhosts']==25610 and plan['source_changes']==[]
 while True:
  assert not stop();prior=json.loads((P/'PTO_CURRENT_EXPANSION.json').read_text())
  if prior['status']=='all22_exact_original_PTO_APIs_fullchecker15raw_guard_valid':assert prior['accepted_original_APIs']==22;break
  assert prior['status'] in ('running','resource_deferred','CPU_deployment'),'PTOpredecessor fault needsinspection, do not hide failure'
  time.sleep(10)
 owns_current=True;c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'qualified_H2_buildprofile_original_ExpReciprocal_H1_host_then_device_build','durable_log':'native-math-build-queued.log','snapshot_read_only':True})
 for stage,entry in [('device','build-device-entry.sh')]:
  while True:
   assert not stop()
   with (P/'controller.lock').open('a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX);okay,gate=c.admit()
    if okay:
     tag='server1-native-math-'+stage+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');receipt=N+'/'+tag+'-guard.json';c.put(A/(tag+'.GATE.json'),gate);s.update(status='guarded_original_'+stage+'_build',tag=tag,started_at=now());c.put(P/'OPS_NN_MATH_NATIVE_BUILD.json',s)
     cmd='source '+ROOT+'/environment.sh; exec '+JOB+' '+c.R+'/opportunistic_guard.py run --owner root/baseline-native-math/'+tag+' --wait 10 --timeout 600 --receipt '+receipt+' -- bash '+N+'/'+entry
     with (A/(tag+'.log')).open('w') as out:rc=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
     got=c.remote('cat '+receipt,stdout=subprocess.PIPE,check=True);raw=got.stdout;guard=json.loads(raw);(A/(tag+'-guard.json')).write_bytes(raw)
    else:s.update(status='resource_deferred',stage=stage,checked_at=now());c.put(P/'OPS_NN_MATH_NATIVE_BUILD.json',s)
   if not okay:time.sleep(30);continue
   if rc==75 and guard.get('phase')=='released_without_launch':time.sleep(30);continue
   row={'stage':stage,'tag':tag,'exit_code':rc,'guard':guard,'log':tag+'.log','dual_archive':[]}
   for f in [A/(tag+'.log'),A/(tag+'-guard.json')]:
    dest=N+'/'+f.name;code='from pathlib import Path;import sys;p=Path('+repr(dest)+');b=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==b;p.write_bytes(b)';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();row['dual_archive'].append({'remote':dest,'sha256':h})
   s['stages'].append(row);c.put(P/'OPS_NN_MATH_NATIVE_BUILD.json',s);assert rc==0 and guard.get('performance_valid') is True and guard.get('workload_exit_code')==0,'Build/guard fault retained; do not blindretry'
   break
 s['status']='both_original_build_stages_done_manifest_assembly_and_runtime_qualification_pending'
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:
 s['checked_at']=now();c.put(P/'OPS_NN_MATH_NATIVE_BUILD.json',s)
 if owns_current:c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'next_action':'Manifest originalhost/API plusalloriginaldeviceobjects/catalog, assembleprivatepackage, verifyactualprivatekernel runtime beforeExpReciprocal10formal. Buildnotcoverage.','snapshot_read_only':True})
