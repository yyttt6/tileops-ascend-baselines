"""CPU-only input/source qualification, doesnot own CURRENT or touchNPU."""
from pathlib import Path
import json,datetime,subprocess,shlex,hashlib,fcntl,sys,os
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
lock=(P/'native-dequant_foreach-inputs.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ids=[512, 513, 514, 515, 516, 517, 518];state={'pid':os.getpid(),'entry':Path(__file__).name,'queued_at':now(),'status':'CPU_only_missing_inputs','scope':ids,'zero_NPU_or_baseline_coverage':True,'counts_as_primary_dispatch':False}
def stop():assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
try:
 stop();prior=json.loads((P/'PTO_CURRENT_EXPANSION.json').read_text());assert prior['accepted_original_APIs']==22
 records={}
 for name in ['ops_nn_dequant_foreach_cached_fidelity.py','verify_ops_nn_dequant_foreach_cached.py']:
  p=P/name;dst=c.R+'/'+name;code='from pathlib import Path;import sys;Path('+repr(dst)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=p.read_bytes(),check=True);h=c.remote('sha256sum '+dst,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0];assert h==hashlib.sha256(p.read_bytes()).hexdigest();records[name]=h
 query='import pathlib,json;r=pathlib.Path('+repr(c.R)+');print(json.dumps([i for i in '+repr(ids)+' if not (r/"prepared-rollout"/str(i)/"prepare.json").exists()]))';missing=json.loads(c.remote('python3 -c '+shlex.quote(query),stdout=subprocess.PIPE,text=True,check=True).stdout);state.update(missing_inputs=missing,deployed_helpers=records);c.put(P/'OPS_NN_DEQUANT_FOREACH_INPUTS.json',state)
 for j in range(0,len(missing),2):
  stop();batch=missing[j:j+2];cmd='TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+c.R+'/prepare_rollout.py native-dequant_foreach-next '+' '.join(map(str,batch))
  with (P/f'native-dequant_foreach-inputs-{j}.CPU.log').open('w') as out:rc=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'CPUinput fault retained beforeNPU'
 for i in ids:
  stop()
  with (P/f'native-dequant_foreach-{i}-source.CPU.log').open('w') as out:rc=c.remote('python3 '+c.R+'/verify_ops_nn_dequant_foreach_cached.py '+str(i),stdout=out,stderr=subprocess.STDOUT).returncode
  assert rc==0,'Frozen source/payload fault, no deviceattempt'
 state.update(status='all7_CPU_inputs_source_ready_runtime_qualification_pending',verified_inputs=len(ids))
except BaseException as e:state.update(status='CPU_source_input_fault_requires_diagnosis',error=repr(e));raise
finally:state['checked_at']=now();c.put(P/'OPS_NN_DEQUANT_FOREACH_INPUTS.json',state)
