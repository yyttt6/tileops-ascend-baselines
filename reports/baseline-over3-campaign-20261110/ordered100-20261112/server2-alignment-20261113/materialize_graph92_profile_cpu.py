from pathlib import Path
import shutil,json,hashlib,datetime
base=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native');src=base/'graph92-complete-profile-v1/sdk';root=base/'graph92-complete-profile-v2';assert not root.exists();root.mkdir();dst=root/'sdk'
shutil.copytree(src,dst,symlinks=True,ignore=lambda path,names:['lib64'] if Path(path)==src/'aarch64-linux' else [])
shutil.copytree(src/'aarch64-linux/lib64',dst/'aarch64-linux/lib64',symlinks=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={}
for p in (dst/'aarch64-linux/lib64').rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(dst);old=src/rel;h=sha(p);assert h==sha(old),str(rel);assert not p.is_symlink();files[str(rel)]={'sha256':h,'bytes':p.stat().st_size,'original_source':str(old.resolve())}
v={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'private_sdk':str(dst),'files':files,'bytes':sum(x['bytes'] for x in files.values()),'reason':'Graph GetSelfLibraryDir resolves symlink to original SDK; matching original92 TBE/AOE libs were invisible at original SDK absolute path. Materialize exactsamebytes only in new privateprefix, no sharedmutations/source/math/flags/checker edits','no_runtime_acceptance':True};(root/'PROVENANCE.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
