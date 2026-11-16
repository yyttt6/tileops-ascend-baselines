from pathlib import Path
import sys,json,datetime,fcntl,subprocess,hashlib,shlex
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';sys.path.insert(0,str(D));import rollout_controller as c
N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-pool-indices-v1';ROOT='/opt/baseline-over3-20261110/server2-aligned-stack';assert not (D/'STOP').exists() and not (D.parent/'STOP').exists();tag='pool-indices-tiling-registry-v1-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f');owner='root/baseline-native-diagnosis/'+tag;receipt=N+'/'+tag+'-guard.json'
with (D/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);okay,gate=c.admit();assert okay,('Freshgate rejected; no diagnosticdevice import',gate);c.put(A/(tag+'.GATE.json'),gate)
 entry='source '+ROOT+'/environment.sh; source '+N+'/compile-entry-env-registration-v2.sh; exec '+ROOT+'/job-python-v1/bin/python '+c.R+'/opportunistic_guard.py run --owner '+owner+' --wait 10 --timeout 120 --receipt '+receipt+' -- '+ROOT+'/job-python-v1/bin/python '+N+'/probe_pool_indices_tiling_registry.py '+owner+' '+tag+'.json'
 with (A/(tag+'.log')).open('w') as out:r=c.remote(entry,stdout=out,stderr=subprocess.STDOUT)
for suffix in ['-guard.json','.json','.json.maps']:
 raw=c.remote('cat '+N+'/'+tag+suffix,stdout=subprocess.PIPE,check=True).stdout;(A/(tag+suffix)).write_bytes(raw)
c.put(A/'POOL_INDICES_TILING_REGISTRY_DIAGNOSIS.json',{'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tag':tag,'exit_code':r.returncode,'not_native_or_compiler_acceptance':True,'result':json.loads((A/(tag+'.json')).read_text())});print((A/(tag+'.json')).read_text())
