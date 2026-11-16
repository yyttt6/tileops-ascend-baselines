from pathlib import Path
import json,shutil,hashlib,os,datetime
base=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1');root=base.parent/'ops-nn-unary-gather-v1';assert not root.exists();assert shutil.disk_usage('/').free>=20000000000
m=json.loads((base/'SOURCE_MANIFEST.json').read_text());dst=root/'source';dst.mkdir(parents=True)
for name,record in m['files'].items():
 src=base/'source'/name;out=dst/name;out.parent.mkdir(parents=True,exist_ok=True)
 assert hashlib.sha256(src.read_bytes()).hexdigest()==record['sha256']
 if src.is_symlink():out.symlink_to(os.readlink(src))
 else:shutil.copy2(src,out)
 assert hashlib.sha256(out.read_bytes()).hexdigest()==record['sha256']
(root/'SOURCE_MANIFEST.json').write_bytes((base/'SOURCE_MANIFEST.json').read_bytes());(root/'dependencies').symlink_to(base/'dependencies');(root/'compat').symlink_to(base/'compat')
r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'source_files_verified':len(m['files']),'source_pin':m['pin'],'source_patch_sha256':m['patch_sha256'],'original_source_edited':False,'selected_ops':['gather_elements_v2'],'dependency_and_header_roots':'Previouslyverifiedoriginaltaskprivate roots via exactsymlink, no SDK/sharedruntime modifications','source_manifest_sha256':hashlib.sha256((root/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()};(root/'SOURCE_PREPARATION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
