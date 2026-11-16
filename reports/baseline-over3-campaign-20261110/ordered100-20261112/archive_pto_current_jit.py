"""CPU archive of completed ownrun JIT payloads; never migrates/deletes data orusesNPU."""
from pathlib import Path
import json,subprocess,shlex,hashlib,tarfile,datetime
P=Path(__file__).resolve().parent;H='openlibing-DevEnv-661495';R='/opt/baseline-over3-20261110/ordered100-20261112'
def archive(cp):
 e=json.loads(cp.read_text());tag=e['tag'];done=P/(tag+'-JIT_ARCHIVE.json')
 if done.exists():return json.loads(done.read_text())
 valid=[x for x in e['batch']['results'] if x['status']=='raw_checked_pending_guard']
 if not valid:return None
 assert e['guard']['performance_valid'] is True
 manifests=[json.loads((P/'remote/runs'/tag/str(x['ordinal'])/'PTO_JIT_ARTIFACTS.json').read_text()) for x in valid];files={k:v for m in manifests for k,v in m.items()};assert files
 name=tag+'-jit-payloads.tar.gz';root=R+'/native/pto-registration-current-v1/runs/'+tag
 code=r'''from pathlib import Path
import sys,json,hashlib,subprocess,tempfile
root=Path(ROOT);dst=Path(DEST);assert not dst.exists();files=json.load(sys.stdin);names=[]
for p,h in files.items():
 p=Path(p);assert p.is_relative_to(root) and p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h;names.append(str(p.relative_to(root)))
with tempfile.NamedTemporaryFile(mode='wb') as f:
 f.write(b'\0'.join(x.encode() for x in names)+b'\0');f.flush();subprocess.run(['tar','--dereference','-C',str(root),'-I','gzip -1','-cf',str(dst),'--null','-T',f.name],check=True)
print(hashlib.sha256(dst.read_bytes()).hexdigest())
'''.replace('ROOT',repr(root)).replace('DEST',repr(R+'/'+name))
 r=subprocess.run(['ssh',H,'python3 -c '+shlex.quote(code)],input=json.dumps(files),stdout=subprocess.PIPE,text=True,check=True);digest=r.stdout.strip()
 with (P/name).open('wb') as f:subprocess.run(['ssh',H,'cat '+shlex.quote(R+'/'+name)],stdout=f,check=True)
 assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 with tarfile.open(P/name) as t:
  members={m.name:m for m in t.getmembers()};assert len(members)==len(files)
  for path,h in files.items():
   relative=str(Path(path).relative_to(root));assert hashlib.sha256(t.extractfile(members[relative]).read()).hexdigest()==h
 proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tag':tag,'artifacts':len(files),'archive':name,'archive_sha256':digest,'remote_archive':R+'/'+name,'all_recorded_artifact_bytes_verified_in_bothcopies':True,'no_NPU_no_deletion_no_newcoverage':True};done.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof));return proof
if __name__=='__main__':
 for cp in sorted(P.glob('rollout-pto-current-*.CHECKPOINT.json')):archive(cp)
