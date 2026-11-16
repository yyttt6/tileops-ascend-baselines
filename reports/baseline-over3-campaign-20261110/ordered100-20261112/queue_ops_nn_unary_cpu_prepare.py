"""Only original CPU inputs/source qualification; rejects NPU imports in prepare."""
import pathlib,json,subprocess,sys,fcntl,datetime,hashlib,shlex,os
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists()
lock=(P/'ops-nn-unary-cpu-prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'entry':pathlib.Path(__file__).name,'queued_at':now(),'status':'CPU_prepare_source_and_missing_inputs','scope':list(range(385,390))+list(range(611,616)),'no_NPU_work_authorized_by_this_controller':True,'zero_baseline_coverage_from_preparation':True}
c.put(P/'OPS_NN_UNARY_INPUTS.json',state)
try:
 assert not stop();prior=json.loads((P/'OPS_NN_NATIVE92_REMAINING.json').read_text());assert prior['status']=='all13_remaining_exact_original_APIs_fullchecker15raw_guard_valid' and prior['accepted_original_APIs']==16
 assert not pathlib.Path('/proc/'+str(prior['pid'])+'/cmdline').exists()
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':pathlib.Path(__file__).name,'updated_at':now(),'phase':'CPU_only_frozen_unary_inputs_source_qualification_while_H2nativebuild','durable_log':'ops-nn-unary-cpu-prepare.log','snapshot_read_only':True})
 records={}
 for name in ['ops_nn_unary_cached_fidelity.py','verify_ops_nn_unary_cached.py']:
  f=P/name;dst=c.R+'/'+name;code='import pathlib,sys;pathlib.Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=f.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(f.read_bytes()).hexdigest();records[name]=h
 state['deployed_helpers']=records;scope=state['scope'];query='import pathlib,json;r=pathlib.Path('+repr(c.R)+');ids='+repr(scope)+';print(json.dumps([i for i in ids if not (r/"prepared-rollout"/str(i)/"prepare.json").exists()]))';missing=json.loads(c.remote('python3 -c '+shlex.quote(query),stdout=subprocess.PIPE,text=True,check=True).stdout);state['missing_inputs']=missing
 for j in range(0,len(missing),2):
  assert not stop();ids=missing[j:j+2];cmd='TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+c.R+'/prepare_rollout.py native-unary-next '+' '.join(map(str,ids))
  with (P/('native-unary-inputs-'+str(j)+'.CPU.log')).open('w') as out:rc=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'CPUinput/source failure; preserve andstop beforeanyNPU'
 for i in scope:
  assert not stop()
  with (P/f'native-unary-{i}-source.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_ops_nn_unary_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'Source/payload identity fault; no numerical retries'
 state.update(status='CPU_inputs_source_ready_native_API_symbols_and_device_qualification_pending',verified_inputs=len(scope))
except BaseException as e:state.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:
 state['checked_at']=now();c.put(P/'OPS_NN_UNARY_INPUTS.json',state);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':pathlib.Path(__file__).name,'checked_at':now(),'outcome':state['status'],'next_action':'H2originalUnaryhostbuild support terminal needsoriginalABIexports/prototypes qualification then originaldevicebuild/probes. CPUinputreadiness isnotbaselineacceptance. H1otherqualifiedwork independent.','snapshot_read_only':True})
