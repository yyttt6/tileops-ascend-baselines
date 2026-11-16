import os,pathlib,hashlib,json,subprocess,datetime
root=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1');sdk=pathlib.Path(os.environ['ASCEND_HOME_PATH']);compat=root/'compat';(compat/'include/ascendc').mkdir(parents=True,exist_ok=True)
links=[(compat/'include/ascendc/host_api',sdk/'aarch64-linux/asc/include'),(compat/'include/tiling',sdk/'aarch64-linux/asc/include/tiling'),(compat/'ascendc',sdk/'aarch64-linux/ascendc')]
records=[]
for link,target in links:
 assert target.is_dir(),str(target)
 if link.is_symlink():assert link.resolve()==target.resolve()
 else:assert not link.exists();link.symlink_to(target,target_is_directory=True)
 records.append({'link':str(link),'target':str(target)})
header=compat/'include/ascendc/host_api/tiling/template_argument.h';canonical=sdk/'aarch64-linux/asc/include/utils/tiling/template_argument.h';assert header.resolve()==canonical.resolve()
cmd=[os.environ['CXX'],'-x','c++','-std=c++17','-fsyntax-only','-I'+str(compat/'include'),'-I'+str(sdk/'aarch64-linux/include'),'-']
r=subprocess.run(cmd,input='#include <ascendc/host_api/tiling/template_argument.h>\nint main(){return 0;}\n',stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
result={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'projection':records,'canonical_header_sha256':hashlib.file_digest(canonical.open('rb'),'sha256').hexdigest(),'source_byte_changes':False,'SDK_tree_changes':False,'host_header_compile_command':cmd,'host_header_compile_exit_code':r.returncode,'diagnostic':r.stderr,'CPU_only_fsyntax_no_link_no_execution':True}
(root/'SDK_INCLUDE_PROJECTION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
