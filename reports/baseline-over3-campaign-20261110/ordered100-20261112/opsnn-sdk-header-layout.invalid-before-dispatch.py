import pathlib,os,json,hashlib
sdk=pathlib.Path('/opt/baseline-over3-20261110/unified-stack-v1/toolchain-candidate/sdk-9.2.0')
links=[(sdk/'aarch64-linux/include/tiling','../asc/include/tiling'),(sdk/'aarch64-linux/include/ascendc/host_api','../../asc/include')]
records=[]
for link,target in links:
 resolved=link.parent/target;assert resolved.is_dir(),str(resolved)
 if link.is_symlink():assert os.readlink(link)==target
 else:assert not link.exists(),str(link);link.symlink_to(target,target_is_directory=True)
 records.append({'link':str(link),'target':target,'scope':'task-private '/opt/baseline-over3-20261110/unified-stack-v1/toolchain-candidate/sdk-9.2.0' sameversion headers; no source-byte changes'})
requested=sdk/'aarch64-linux/include/ascendc/host_api/tiling/template_argument.h';canonical=sdk/'aarch64-linux/asc/include/utils/tiling/template_argument.h';assert requested.resolve()==canonical.resolve()
assert (sdk/'aarch64-linux/include/tiling/tiling_api.h').is_file()
proof={'links':records,'requested_header':str(requested),'canonical_header':str(canonical),'canonical_sha256':hashlib.file_digest(canonical.open('rb'),'sha256').hexdigest(),'kernel_or_header_bytes_changed':False,'shared_'/opt/baseline-over3-20261110/unified-stack-v1/toolchain-candidate/sdk-9.2.0'_untouched':True}
(pathlib.Path('/opt/baseline-over3-20261110/ordered100-20261112/native-rebuild/ops-nn-v1/SDK_HEADER_LAYOUT_REPAIR.json')).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
