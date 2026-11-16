"""Bounded CPU-only build; independent perhost, no original large timeout rerun."""
from pathlib import Path
import os,sys,json,hashlib,datetime,subprocess,shlex,fcntl,concurrent.futures
P=Path(__file__).resolve().parent;N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-tensor-trunc-host-tile-v1';A=P/'server2-alignment-20261113';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();lock=(P/'trunc-host-tile-build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s={'pid':os.getpid(),'entry':Path(__file__).name,'queued_at':now(),'status':'CPU_only_original_vendor_source_and_host_argument_review','scope':[733,734],'accepted_735_not_repeated':True,'no_native_work_or_coverage_claim':True}
def put():s['checked_at']=now();(P/'TRUNC_HOST_TILE_BUILD.json').write_text(json.dumps(s,indent=2)+'\n')
def stop():return (P/'STOP').exists() or (P.parent/'STOP').exists() or json.loads((P.parent/'STATE.json').read_text())['status']!='active'
put()
try:
 assert not stop();(P/'CURRENT_PROCESS.json').write_text(json.dumps({'status':'active','pid':os.getpid(),'entry':Path(__file__).name,'updated_at':now(),'phase':'CPU_only_Trunc_host_API_argument_batching_compile_not_native_or_coverage','snapshot_read_only':True},indent=2)+'\n');row=json.loads((P/'OPS_TENSOR_TRUNC_CURRENT.json').read_text())['rows'][0];sources={k:v for k,v in row['selected_sources'].items() if '/ascend_baselines/ops-tensor/' in k};assert sources
 def one(pair):
  n,h=pair;assert not stop();cmd=['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8',h]
  for name,raw in [('build_trunc_host_tile_cpu.py',(P/'build_trunc_host_tile_cpu.py').read_bytes()),('ACCEPTED_735_VENDOR_SOURCE_SHA.json',json.dumps(sources,indent=2).encode())]:
   path=N+'/'+name;code='from pathlib import Path;import sys;p=Path('+repr(path)+');p.parent.mkdir(parents=True,exist_ok=True);raw=sys.stdin.buffer.read();assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw)';subprocess.run(cmd+['python3 -c '+shlex.quote(code)],input=raw,check=True)
  with (A/f'TRUNC_HOST_TILE_BUILD_HOST{n}.log').open('w') as out:r=subprocess.run(cmd+['source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh; exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+N+'/build_trunc_host_tile_cpu.py'],stdout=out,stderr=subprocess.STDOUT)
  got=subprocess.run(cmd+['cat '+N+'/BUILD_RECEIPT.json'],capture_output=True);assert got.returncode==0;f=A/f'TRUNC_HOST_TILE_BUILD_HOST{n}.json';f.write_bytes(got.stdout);proof=json.loads(got.stdout);assert r.returncode==0 and proof['exit_code']==0;return {'host':h,'record':str(f.relative_to(P)),'binary_sha256':proof['binary_sha256']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:s['hosts']=list(pool.map(one,[(1,'openlibing-DevEnv-661495'),(2,'openlibing-DevEnv-055409')]))
 s['status']='CPU_compiled_requires_bounded_UB_and_full_API_functional_qualification'
except BaseException as e:s.update(status='stopped_requires_diagnosis',error=repr(e));raise
finally:put();(P/'CURRENT_PROCESS.json').write_text(json.dumps({'status':'group_finished_next_dispatch_required','pid':None,'prior_pid':os.getpid(),'entry':Path(__file__).name,'checked_at':now(),'outcome':s['status'],'snapshot_read_only':True},indent=2)+'\n')
