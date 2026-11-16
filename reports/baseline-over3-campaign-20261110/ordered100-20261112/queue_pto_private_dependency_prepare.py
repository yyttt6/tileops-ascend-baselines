"""CPU-only exact server2 locked SDK dependency deployment, never uses NPU."""
from pathlib import Path
import json,datetime,hashlib,subprocess,shlex,fcntl,os
D=Path(__file__).resolve().parent;H2='openlibing-DevEnv-055409';H1='openlibing-DevEnv-661495';SRC='/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2';DST='/opt/baseline-over3-20261110/ordered100-20261112/native/pto-locked-sdk-920b2-v1'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
lock=(D/'pto-dependency-prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s={'pid':os.getpid(),'entry':Path(__file__).name,'queued_at':now(),'status':'CPU_source_inventory','no_NPU_or_baseline_coverage':True,'source_host':H2,'destination_host':H1,'source':SRC,'destination':DST,'not_cold_data_migration':True,'source_retained_unchanged':True}
def put():
 p=D/'PTO_PRIVATE_DEPENDENCY_PREPARE.json';t=p.with_suffix('.cpu.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
def run(host,cmd,**kw):return subprocess.run(['ssh',host,cmd],**kw)
put()
try:
 assert not (D/'STOP').exists() and not (D.parent/'STOP').exists()
 code=r'''from pathlib import Path
import json,hashlib,os,sys,shutil
r=Path(SRC);assert shutil.disk_usage('/').free>=20000000000
rows={}
for p in sorted(r.rglob('*')):
 if p.is_symlink():rows[str(p.relative_to(r))]={'symlink':os.readlink(p)}
 elif p.is_file():rows[str(p.relative_to(r))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'mode':p.stat().st_mode&511}
print(json.dumps({'root':str(r),'files':rows},sort_keys=True))
'''.replace('SRC',repr(SRC))
 r=run(H2,'python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,check=True);manifest=D/'PTO_SDK920B2_SERVER2_SOURCE_MANIFEST.json';assert not manifest.exists();manifest.write_bytes(r.stdout)
 s.update(status='CPU_dependency_archive',manifest_sha256=hashlib.sha256(r.stdout).hexdigest());put();archive=D/'PTO_SDK920B2_SOURCE.tar.gz';assert not archive.exists()
 with archive.open('wb') as f:run(H2,'tar -C '+shlex.quote(SRC)+' -I '+shlex.quote('gzip -1')+' -cf - .',stdout=f,check=True)
 s.update(status='CPU_private_deployment',archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest());put()
 # Only necessary build dependencies are copied; no source deletion/symlink migration/shared-stack edits.
 cmd='test ! -e '+shlex.quote(DST)+' && mkdir -p '+shlex.quote(DST)+' && tar -C '+shlex.quote(DST)+' -xzf -'
 with archive.open('rb') as f:run(H1,cmd,stdin=f,check=True)
 code=r'''from pathlib import Path
import sys,json,hashlib,os
m=json.load(sys.stdin);r=Path(DST);assert set(m['files'])=={str(p.relative_to(r)) for p in r.rglob('*') if p.is_file() or p.is_symlink()}
for name,x in m['files'].items():
 p=r/name
 if 'symlink' in x:assert p.is_symlink() and os.readlink(p)==x['symlink'],name
 else:assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'] and p.stat().st_mode&511==x['mode'],name
print(json.dumps({'all_regular_file_bytes_and_symlink_texts_verified':True,'entries':len(m['files']),'no_NPU':True}))
'''.replace('DST',repr(DST));r=run(H1,'python3 -c '+shlex.quote(code),input=manifest.read_bytes(),stdout=subprocess.PIPE,check=True);v=json.loads(r.stdout);s.update(status='all_original_SDK_files_verified_private_relocation_only_runtime_pending',verification=v,checked_at=now());put()
except BaseException as e:s.update(status='CPU_dependency_fault_requires_diagnosis',error=repr(e));put();raise
finally:s['checked_at']=now();put()
