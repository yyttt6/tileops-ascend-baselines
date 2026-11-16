from pathlib import Path
import json,hashlib,subprocess,shutil,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-chunk-v1');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((N/'FLA_ORIGINAL_SOURCE_MANIFEST.json').read_text());assert all(sha(N/'source'/n)==h for n,h in m['files'].items())
candidates=[p for p in (N/'source/build').rglob('vendors/fla_npu_transformer') if (p/'op_api/lib/libcust_opapi.so').is_file() and '_CPack_Packages' in p.parts];assert len(candidates)==1,candidates
original=candidates[0];binary=original/'op_api/lib/libcust_opapi.so';exports=subprocess.check_output(['nm','-D','--defined-only',str(binary)],text=True);assert all(x in exports for x in ['aclnnChunkFwdHGetWorkspaceSize','aclnnChunkFwdH','aclnnChunkFwdO'])
assert binary.read_bytes()[18:20]==bytes([183,0]),'Required actual AArch64 host binary'
target=N/'python_binary_aot_v4/fla_npu';assert not target.exists(),'Preserve existing package, do not overwrite'
shutil.copytree(N/'source/torch_custom/fla_npu/fla_npu',target,ignore=shutil.ignore_patterns('opp','__pycache__','*.pyc'));shutil.copytree(original,target/'opp/vendors/fla_npu_transformer',symlinks=True)
device_bins=[p for p in target.rglob('*.o') if '/kernel/ascend950/' in str(p)];assert device_bins,'Original precompiled native device binaries required'
files={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()};out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files_verified':len(m['files']),'source_manifest_sha256':sha(N/'FLA_ORIGINAL_SOURCE_MANIFEST.json'),'original_pin':m['source_pin'],'package':str(target),'artifact_sha256':files,'original_unmodified_950_AoT_build':True,'device_validation':'pending full checker/15raw/sourceELF/guard; build is not coverage'};(N/'PACKAGE_AOT_V4_MANIFEST.json').write_text(json.dumps(out,indent=2)+chr(10));print(json.dumps({'package_files':len(files),'device_validation':'pending'}))
