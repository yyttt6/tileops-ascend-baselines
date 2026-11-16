"""Only accepted inactive payload copies; remote fullPT+local checked gzip then symlink."""
import pathlib,json,subprocess,hashlib,gzip,fcntl,shlex
if (pathlib.Path(__file__).resolve().parent/'NO_OWN_MIGRATION').exists():raise SystemExit('User disabled own migration; another agent owns storage work')
P=pathlib.Path(__file__).resolve().parent;H='openlibing-DevEnv-661495';R='/opt/baseline-over3-20261110/ordered100-20261112';lock=(P/'disk-recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(f):
 h=hashlib.sha256()
 while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
ledger=json.loads((P/'LEDGER.json').read_text());roll=json.loads((P/'ROLLOUT_LEDGER.json').read_text());allow=[]
for folder,entries in [('prepared',ledger),('prepared-rollout',roll)]:
 for key,e in entries.items():
  if e.get('result',{}).get('status')!='raw_checked_pending_guard':continue
  meta=P/'remote'/folder/key/'prepare.json'
  if meta.exists():allow.append({'folder':folder,'ordinal':int(key),'sha256':json.loads(meta.read_text())['payload_sha256']})
code='import pathlib,json;rows='+repr(allow)+';r=pathlib.Path('+repr(R)+');out=[]\nfor x in rows:\n p=r/x["folder"]/str(x["ordinal"])/"payload.pt"\n if p.exists() and not p.is_symlink():out.append(dict(x,bytes=p.stat().st_size))\nprint(json.dumps(sorted(out,key=lambda x:-x["bytes"])))'
items=json.loads(subprocess.check_output(['ssh',H,'python3 -c '+shlex.quote(code)],text=True));receipts=P/'DISK_RECOVERY_RECEIPT.json';done=json.loads(receipts.read_text()) if receipts.exists() else [];moved=0
(P/'DISK_RECOVERY_ALLOWLIST.json').write_text(json.dumps(items,indent=2))
for x in items:
 if moved>=8*1024**3:break
 src=R+'/'+x['folder']+'/'+str(x['ordinal'])+'/payload.pt';dst='/workspace/user_data/baseline-over3-20261110/completed-ordered-inputs/'+x['folder']+'/'+str(x['ordinal'])+'/payload.pt';expected=x['sha256'];local=P/'completed-inputs'/(x['folder']+'-'+str(x['ordinal'])+'.pt.gz');local.parent.mkdir(exist_ok=True)
 code='''import pathlib,shutil,hashlib,json
s=pathlib.Path(%r);d=pathlib.Path(%r)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
assert digest(s)==%r;d.parent.mkdir(parents=True,exist_ok=True)
if not d.exists():shutil.copyfile(s,d)
assert digest(d)==%r
print(json.dumps({'bytes':d.stat().st_size}))
'''%(src,dst,expected,expected)
 info=json.loads(subprocess.check_output(['ssh',H,'python3 -c '+shlex.quote(code)],text=True))
 if not local.exists():
  tmp=local.with_suffix('.partial')
  with tmp.open('wb') as f:subprocess.run(['ssh',H,'gzip -1 -c '+shlex.quote(dst)],stdout=f,check=True)
  with gzip.open(tmp,'rb') as f:assert sha(f)==expected
  tmp.replace(local)
 with gzip.open(local,'rb') as f:assert sha(f)==expected
 code='''import pathlib,hashlib,os
s=pathlib.Path(%r);d=pathlib.Path(%r)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
assert digest(s)==%r and digest(d)==%r
t=s.with_name('payload.pt.migration-link');t.symlink_to(d);os.replace(t,s)
'''%(src,dst,expected,expected)
 subprocess.run(['ssh',H,'python3 -c '+shlex.quote(code)],check=True)
 with local.open('rb') as f:compressed=sha(f)
 done.append(dict(x,original=src,target=dst,local_gzip=str(local),local_gzip_sha256=compressed));tmp=receipts.with_suffix('.tmp');tmp.write_text(json.dumps(done,indent=2));tmp.replace(receipts);moved+=info['bytes'];print('MIGRATED',x['folder'],x['ordinal'],info['bytes'],'total',moved,flush=True)
print('DONE freed',moved,flush=True)
