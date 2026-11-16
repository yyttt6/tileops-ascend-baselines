"""Only reclaim audited per-attempt expanded profiles; preserve both full archives."""
from pathlib import Path
import json,hashlib,subprocess,tarfile
P=Path(__file__).resolve().parent
H='openlibing-DevEnv-661495'
import fcntl
lock=(P/'current-profile-compaction.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
allow=[]
for sub in ['vllm-rms-api-final-server1','vllm-rms-expand-server1','vllm-gemm-fp16-server1','vllm-fused-server1','vllm-neg-server1','vllm-cnn-server1','vllm-fused5120-server1','vllm-ffn-server1','vllm-silu-server1']:
 d=P/sub
 for r in json.loads((d/'RESULTS.json').read_text()):
  if r['status'] not in ['formal_raw_verified','pilot_raw_verified']:continue
  mode='formal' if r['status']=='formal_raw_verified' else 'pilot'
  name=f"{mode}-{r['index']}-{r['variant']}-{r['attempt']}.tar.gz"
  archive=d/name
  if not archive.exists():continue
  rf=d/'remote/runs'/str(r['index'])/r['variant']/r['attempt']/'result.json'
  assert hashlib.sha256(rf.read_bytes()).hexdigest()==r['result_sha256']
  digest=hashlib.sha256(archive.read_bytes()).hexdigest()
  root='/opt/baseline-over3-20261110/'+sub
  allow.append(dict(root=root,archive=name,sha256=digest,profile=f"runs/{r['index']}/{r['variant']}/{r['attempt']}/profiles",result_sha256=r['result_sha256']))
(P/'CURRENT_PROFILE_COMPACTION_ALLOWLIST.json').write_text(json.dumps(allow,indent=2)+'\n')
code=r'''
import sys,json,pathlib,tarfile,hashlib
rows=json.loads(sys.stdin.readline());done=[]
def digest_stream(f):
 h=hashlib.sha256()
 while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
for r in rows:
 root=pathlib.Path(r['root']);assert root.parent==pathlib.Path('/opt/baseline-over3-20261110')
 profile=root/r['profile'];archive=root/r['archive']
 if not profile.exists():continue
 assert not profile.is_symlink()
 with archive.open('rb') as f:assert digest_stream(f)==r['sha256']
 files=[x for x in profile.rglob('*') if x.is_file()];checked=[]
 with tarfile.open(archive) as t:
  members={m.name:m for m in t.getmembers() if m.isfile()}
  for f in files:
   assert not f.is_symlink() and f.resolve().is_relative_to(profile.resolve())
   rel=str(f.relative_to(root));m=members[rel]
   with f.open('rb') as stream:actual=digest_stream(stream)
   with t.extractfile(m) as stream:assert actual==digest_stream(stream)
   checked.append((f,actual,f.stat().st_size))
 for f,sha,size in checked:
  with f.open('rb') as stream:assert digest_stream(stream)==sha
  f.unlink()
 for folder in sorted([x for x in profile.rglob('*') if x.is_dir()],key=lambda x:len(x.parts),reverse=True):folder.rmdir()
 profile.rmdir();done.append(dict(profile=str(profile),archive_sha256=r['sha256'],files=len(checked),bytes=sum(x[2] for x in checked)))
 print(json.dumps(done[-1]),flush=True)
'''
# Structured data on stdin; code is passed as one SSH-quoted argument.
import shlex
proc=subprocess.run(['ssh',H,'python3 -c '+shlex.quote(code)],input=json.dumps(allow)+'\n',text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(P/'CURRENT_PROFILE_COMPACTION_RECEIPT.jsonl').open('a').write(proc.stdout)
print(proc.stdout);print(proc.stderr);proc.check_returncode()
