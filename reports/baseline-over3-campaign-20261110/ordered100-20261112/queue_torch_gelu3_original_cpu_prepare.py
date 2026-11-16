from pathlib import Path
import os,sys,json,datetime,subprocess,shlex,hashlib,fcntl
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';sys.path.insert(0,str(P));import rollout_controller as c
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();lock=(P/'torch-gelu3-original-prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);f=P/'TORCH_GELU3_CPU_PREPARATION_PROCESS.json';s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_only_source_deploy_prepare','scope':[1361,1362,1363],'queued_at':now(),'zero_formal_coverage':True};c.put(f,s)
try:
 assert json.loads((P.parent/'STATE.json').read_text())['status']=='active' and not (P/'STOP').exists() and not (P.parent/'STOP').exists()
 c.put(P/'CURRENT_PROCESS.json',{'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'CPU_only_exact_published_f037_catalog_and_original_Torch_program_prepare3_no_NPU','snapshot_read_only':True})
 pub=json.loads(Path('/root/dyq/workspace/reports/ascend-web-over10-audit-20261109/baseline-cache.json').read_text());r=[x for x in pub['rows'] if x.get('library')=='torch-eager' and x.get('op')=='GELUAndMulFwdOp'];assert len(r)==3;(P/'TORCH_GELU3_PUBLISHED_ROWS.json').write_text(json.dumps(r,indent=2)+'\n')
 for name in ['prepare_torch_gelu3_original_cpu.py','TORCH_GELU3_PUBLISHED_ROWS.json']:
  p=P/name;code='from pathlib import Path;import sys;Path('+repr(c.R+'/'+name)+').write_bytes(sys.stdin.buffer.read())';c.remote('python3 -c '+shlex.quote(code),input=p.read_bytes(),check=True);assert c.remote('sha256sum '+c.R+'/'+name,stdout=subprocess.PIPE,text=True,check=True).stdout.split()[0]==hashlib.sha256(p.read_bytes()).hexdigest()
 command='source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh; export TORCH_DEVICE_BACKEND_AUTOLOAD=0; exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+c.R+'/prepare_torch_gelu3_original_cpu.py'
 with (P/'torch-gelu3-original-cpu-prepare.log').open('w') as out:rc=c.remote(command,stdout=out,stderr=subprocess.STDOUT).returncode
 assert rc==0,'CPU source/contract failure, no device work attempted';raw=c.remote('cat '+c.R+'/TORCH_GELU3_ORIGINAL_CPU_PREPARATION.json',stdout=subprocess.PIPE,check=True).stdout;(P/'TORCH_GELU3_ORIGINAL_CPU_PREPARATION.json').write_bytes(raw);s.update(status='CPU3_prepared_original_contract_verified_no_NPU_pending_formal',result=json.loads(raw))
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:s['checked_at']=now();c.put(f,s);c.put(P/'CURRENT_PROCESS.json',{'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True})
