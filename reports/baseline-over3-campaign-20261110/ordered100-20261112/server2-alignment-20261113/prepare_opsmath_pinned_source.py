"""CPU-only exact git-pin source export; no library build or NPU imports."""
from pathlib import Path
import subprocess,json,hashlib,datetime,concurrent.futures,shlex
D=Path(__file__).resolve().parent;P=D.parent;repo=Path('/root/dyq/workspace/ascend_baselines/ops-math');pin='64322f6487162c0934ec0364af57ed2f51175a03';N='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1'
assert not (P/'STOP').exists() and not (P.parent/'STOP').exists();assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==pin
arc=D/'OPSMATH_PINNED_SOURCE.tar';assert not arc.exists()
with arc.open('wb') as out:subprocess.run(['git','-C',str(repo),'archive','--format=tar',pin],stdout=out,check=True)
expected=[]
for line in subprocess.check_output(['git','-C',str(repo),'ls-tree','-r','-z',pin]).split(b'\0'):
 if not line:continue
 meta,path=line.split(b'\t',1);mode,kind,blob=meta.decode().split();assert kind=='blob';expected.append((path.decode(),mode,blob))
manifest={'pin':pin,'patch':None,'original_files':expected,'archive_sha256':hashlib.sha256(arc.read_bytes()).hexdigest(),'archive_bytes':arc.stat().st_size,'purpose':'new canonicalH2 ops-math directAdd API build, independent of NN Graph blocker'};mp=D/'OPSMATH_PINNED_SOURCE_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
def deploy(host):
 code='from pathlib import Path;import sys,tarfile,hashlib,json; n=Path('+repr(N)+');assert not n.exists();n.mkdir(parents=True);a=n/"SOURCE.tar";data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()=='+repr(manifest['archive_sha256'])+';a.write_bytes(data);s=n/"source";s.mkdir();t=tarfile.open(a);t.extractall(s,filter="data")'
 with arc.open('rb') as f:subprocess.run(['ssh',host,'/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)],stdin=f,check=True)
 code=r'''from pathlib import Path
import sys,json,hashlib,os,datetime
n=Path(NPATH);m=json.loads(sys.stdin.buffer.read());records={}
for name,mode,blob in m['original_files']:
 p=n/'source'/name;data=os.readlink(p).encode() if mode=='120000' else p.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==blob,name;records[name]={'sha256':hashlib.sha256(data).hexdigest(),'git_blob':blob,'mode':mode}
r={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pin':m['pin'],'patch':None,'files':records,'archive_sha256':m['archive_sha256'],'no_source_changes':True};(n/'SOURCE_MANIFEST.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'root':str(n),'source_files_verified':len(records),'pin':m['pin'],'source_manifest_sha256':hashlib.sha256((n/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()}))
'''.replace('NPATH',repr(N));r=subprocess.check_output(['ssh',host,'/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)],input=mp.read_bytes());out=json.loads(r);(D/('server1' if host.endswith('661495') else 'server2')+'-opsmath-add-source.json') if False else None
 (D/(('server1' if host.endswith('661495') else 'server2')+'-opsmath-add-source.json')).write_bytes(r);return host,out
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for host,result in pool.map(deploy,['openlibing-DevEnv-661495','openlibing-DevEnv-055409']):print(host,result,flush=True)
