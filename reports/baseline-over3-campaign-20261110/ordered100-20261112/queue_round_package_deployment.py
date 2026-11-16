from pathlib import Path
import json,sys,time,os,fcntl,subprocess,datetime
P=Path(__file__).resolve().parent;A=P/'server2-alignment-20261113';os.chdir(P.parents[2])
lock=(P/'round-package-deployment.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def put(x):
 p=P/'ROUND_PACKAGE_PIPELINE.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
s={'pid':os.getpid(),'entry':Path(__file__).name,'status':'CPU_wait_own_original_H2_builds','not_baseline_or_device_claim':True};put(s)
try:
 while True:
  assert not (P/'STOP').exists() and not (P.parent/'STOP').exists()
  h=json.loads((A/'SERVER2_ROUND_REFERENCE_BUILD.json').read_text());d=json.loads((A/'SERVER2_ROUND_DEVICE_BUILD.json').read_text())
  for row in [h,d]:assert row['status'] not in ('stopped_requires_diagnosis','reference_build_failed_requires_diagnosis'),row
  if h['status']==d['status']=='built_pending_identity_and_load':
   assert all(x['exit_code']==0 and x['guard']['performance_valid'] for x in [h,d]);break
  time.sleep(10)
 for name in ['assemble_native_round_package_cpu.py','deploy_round_private_package_cpu.py']:subprocess.run([sys.executable,str(A/name)],check=True)
 m=json.loads((A/'ROUND_UNARY_PACKAGE.v2.server2.json').read_text());header=m['API_headers']['aclnn_foreach_round_off_number.h'];assert 'const aclTensor *roundMode' in header or 'const aclTensor* roundMode' in header,header
 s['status']='original_package_H1_allfiles_verified_prototype_checked_pending_runtime'
except BaseException as e:s.update(status='failed_requires_diagnosis',error=repr(e));raise
finally:s['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();put(s)
