import pathlib,json,hashlib,shutil,datetime,subprocess
N=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1')
old=json.loads((N/'ARTIFACTS-server2-aligned-native-162714998720.json').read_text());dst=N/'complete-private-device-v2';assert not dst.exists();src=pathlib.Path(old['staging_root']);shutil.copytree(src,dst,symlinks=True)
b=N/'source/build-device-v4/binary/ascend950/bin';tbe=dst/'op_impl/ai_core/tbe';shutil.copytree(b/'ascend950/foreach_mul_list',tbe/'kernel/ascend950/foreach_mul_list');config=tbe/'kernel/config/ascend950';config.mkdir(parents=True)
for f in (b/'config/ascend950').glob('foreach_mul_list*.json'):shutil.copy2(f,config/f.name)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file()};assert all(sha(dst/k)==v for k,v in base.items())
m={**old,'staging_root':str(dst),'assembly_checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'assembly_rule':'Original cmake/gen_ops_info.cmake435-450 custom package kernel+config install layout; original scripts unmodified. Original host/JIT1403 files retained byteequal; add original builtin-compiled device binaries/config without editing metadata. Actual custom kernel selection pending.','device_files':{str(p.relative_to(dst)):sha(p) for p in (tbe/'kernel').rglob('*') if p.is_file()},'device_objects':len(list((tbe/'kernel').rglob('*.o'))),'base_artifacts_unchanged':len(base),'private_original_device_kernels_verified':False}
assert m['device_objects']==7
for f in config.glob('*.json'):
 j=json.loads(f.read_text());assert len(j['binList'])>=3
m['all_files']={str(p.relative_to(dst)):sha(p) for p in dst.rglob('*') if p.is_file()}
(N/'COMPLETE_DEVICE_PACKAGE92.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'objects':m['device_objects'],'base_unchanged':m['base_artifacts_unchanged'],'new_files':len(m['device_files']),'manifest_sha256':sha(N/'COMPLETE_DEVICE_PACKAGE92.json')}))
