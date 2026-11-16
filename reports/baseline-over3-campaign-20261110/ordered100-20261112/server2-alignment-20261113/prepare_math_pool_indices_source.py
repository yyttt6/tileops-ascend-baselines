from pathlib import Path
import json,hashlib,os,shutil,datetime
B=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native');old=B/'ops-math-add-v1';V=B/'ops-math-pool-mean-v1';N=B/'ops-math-pool-indices-v1';assert not N.exists();assert shutil.disk_usage('/').free>=20000000000;m=json.loads((old/'SOURCE_MANIFEST.json').read_text());dst=N/'source';dst.mkdir(parents=True)
for name,r in m['files'].items():
 f=old/'source'/name;out=dst/name;out.parent.mkdir(parents=True,exist_ok=True);assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
 if f.is_symlink():out.symlink_to(os.readlink(f))
 else:os.link(f,out)
 assert hashlib.sha256(out.read_bytes()).hexdigest()==r['sha256']
assert len(m['files'])==14279;(N/'SOURCE_MANIFEST.json').write_bytes((old/'SOURCE_MANIFEST.json').read_bytes());(N/'dependencies').symlink_to(old/'dependencies');(dst/'build').symlink_to(N/'build-host-v1')
shutil.copytree(V/'compile-opp',N/'compile-opp',symlinks=True)
# Replace only copied task-local tiling symlinks; source/compiler flags unchanged.
for f in (N/'compile-opp').rglob('*'):
 if f.is_symlink() and str(V) in os.readlink(f):target=os.readlink(f).replace(str(V),str(N));f.unlink();f.symlink_to(target)
s=(V/'generation-entry.sh').read_text().replace(str(V),str(N)).replace('allpairs_math_poolmean','allpairs_math_poolindices').replace('reduce_mean','arg_max_with_value');(N/'generation-entry.sh').write_text(s);r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified_source_files':len(m['files']),'pin':m['pin'],'source_manifest_sha256':hashlib.sha256((N/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),'private_real_source_path':str(dst),'source_build_link':str(N/'build-host-v1'),'original_files_unchanged':True,'host_or_device_build_not_started':True};(N/'SOURCE_PREPARATION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
