from pathlib import Path
import os,json,hashlib,datetime
base=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native');sdk=Path('/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0');root=base/'graph92-complete-profile-v1';assert not root.exists();root.mkdir();dst=root/'sdk';dst.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();links={};additions={}
def link_children(source,target):
 for q in source.iterdir():
  t=target/q.name
  if not t.exists() and not t.is_symlink():t.symlink_to(q,target_is_directory=q.is_dir());links[str(t.relative_to(dst))]=str(q)
link_children(sdk,dst)
def private_dir(p):
 if p==dst:return
 private_dir(p.parent)
 if p.is_symlink():
  old=p.resolve();assert old.is_dir();p.unlink();p.mkdir();link_children(old,p)
 elif not p.exists():p.mkdir()
 assert p.is_dir() and not p.is_symlink()
private_dir(dst/'aarch64-linux/lib64')
# top-level lib64 points at this private lib directory, not canonical SDK.
top=dst/'lib64';assert top.is_symlink();top.unlink();top.symlink_to('aarch64-linux/lib64',target_is_directory=True)
for package in ['graph92-tbe-tik-extracted-v1','graph92-aoe-extracted-v1']:
 src=base/package/'aarch64-linux/lib64'
 for q in src.rglob('*'):
  if not q.is_file():continue
  rel=q.relative_to(src);out=dst/'aarch64-linux/lib64'/rel;private_dir(out.parent)
  if out.exists():assert sha(out)==sha(q),('collision',out,q)
  else:out.symlink_to(q)
  additions[str(out.relative_to(dst))]={'path':str(q),'sha256':sha(q),'bytes':q.stat().st_size}
assert (dst/'aarch64-linux/lib64/libcann_kb.so').is_file()
libs=['libacl_op_compiler.so','libge_compiler.so','libge_common.so','libregister.so']
original={x:sha(sdk/'aarch64-linux/lib64'/x) for x in libs if (sdk/'aarch64-linux/lib64'/x).is_file()}
assert all(sha(dst/'aarch64-linux/lib64'/x)==h for x,h in original.items())
r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'canonical_sdk':str(sdk),'private_sdk':str(dst),'canonical_core_library_sha256_unchanged':original,'new_original92_package_files':additions,'canonical_sdk_mutations':False,'extraction_mode':'--noexec; install and cleanup scripts disabled','runtime_or_device_acceptance':False};(root/'PROVENANCE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
