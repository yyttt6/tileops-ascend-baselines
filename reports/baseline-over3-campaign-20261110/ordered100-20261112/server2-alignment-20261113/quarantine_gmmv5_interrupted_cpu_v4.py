from pathlib import Path
import os,json,hashlib,datetime
N=Path("/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-gmmv5-aot-original-v2");gen=N/'source/build/binary/ascend950/gen';locks=list(gen.glob('*/kernel_meta.lock'));assert len(locks)==8,len(locks)
# Guard owns the host slot; refuse any actual competing build process referencing this private tree.
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  raw=(p/'cmdline').read_bytes().split(b'\0');exe=Path(raw[0].decode()).name if raw else '';cmd=b' '.join(raw).decode(errors='replace')
 except (OSError,IndexError):continue
 if str(N) in cmd and exe in ['bisheng','ccec','gmake','make','opc','python3','python']:
  assert 'opportunistic_guard.py' in cmd or 'quarantine_gmmv5_interrupted_cpu_v4.py' in cmd, (p.name,cmd[:300])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dst=N/'interrupted-meta-preserved-v4';assert not dst.exists();dst.mkdir();records=[]
for lock in locks:
 src=lock.parent;files={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file()};target=dst/src.name;src.rename(target)
 assert all(sha(target/name)==value for name,value in files.items());records.append({'directory':src.name,'lock_contents':(target/'kernel_meta.lock').read_text(),'files':files})
f=N/'INTERRUPTED_META_QUARANTINE_V4.json';f.write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'confirmed terminated own build; SDK compares PID text, does not test actual ownership','preserved':records,'host_device_lock_untouched':True},indent=2));print('PRESERVED_INTERRUPTED_DIRECTORIES',len(records),'proof_sha',sha(f))
