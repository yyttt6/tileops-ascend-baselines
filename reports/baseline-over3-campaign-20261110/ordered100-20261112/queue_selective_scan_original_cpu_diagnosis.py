from pathlib import Path
import os,sys,json,subprocess,shlex,datetime,hashlib,fcntl
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();f=P/'SELECTIVE_SCAN_CPU_DIAGNOSIS_PROCESS.json';lock=(P/'selective-scan-cpu-diagnosis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_only_contract_diagnosis','scope':[1498],'queued_at':now(),'zero_native_qualification':True};c.put(f,s)
try:
 assert json.loads((P.parent/'STATE.json').read_text())['status']=='active' and not (P/'STOP').exists() and not (P.parent/'STOP').exists()
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'CPU_only_scan_complete_published_program_vs_unchanged_frozen_oracle_no_NPU','snapshot_read_only':True})
 file=P/'prove_selective_scan_original_cpu.py';dest=c.R+'/'+file.name;code='from pathlib import Path;import sys;Path('+repr(dest)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=file.read_bytes(),check=True);assert c.remote('sha256sum '+dest,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(file.read_bytes()).hexdigest()
 cmd='source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh; export TORCH_DEVICE_BACKEND_AUTOLOAD=0; exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+dest
 with (P/'selective-scan-original-cpu-diagnosis.log').open('w') as out:rc=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,'CPU diagnosis failed; no native work attempted';raw=c.remote('cat '+c.R+'/SELECTIVE_SCAN_ORIGINAL_CPU_DIAGNOSIS.json',stdout=subprocess.PIPE,check=True).stdout;(P/'SELECTIVE_SCAN_ORIGINAL_CPU_DIAGNOSIS.json').write_bytes(raw);s.update(status='CPU_original_program_vs_frozen_oracle_checked_pending_native_scope_review',result=json.loads(raw))
except BaseException as ex:s.update(status='stopped_requires_diagnosis',error=repr(ex));raise
finally:s['checked_at']=now();c.put(f,s);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
