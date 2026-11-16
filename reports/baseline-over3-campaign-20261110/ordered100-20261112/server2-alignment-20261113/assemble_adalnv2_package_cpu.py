"""CPU-only assembly of original host API and actual950 binaries, no NPU import."""
from pathlib import Path
import json,hashlib,subprocess,shutil,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-adaln-v2');R=N/'source';dest=N/'complete-private-device-v1';assert not dest.exists();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();manifest=json.loads((N/'SOURCE_MANIFEST.json').read_text())
for rel,rec in manifest['files'].items():assert sha(R/rel)==rec['sha256'],rel
libs=list((R/'build_out/_CPack_Packages').rglob('libcust_opapi.so'));assert len(libs)==1;host=libs[0].parents[2];shutil.copytree(host,dest,symlinks=True)
api=dest/'op_api/lib/libcust_opapi.so';exports=subprocess.check_output(['nm','-D','--defined-only',str(api)],text=True);wanted=['aclnnAdaLayerNormV2','aclnnAdaLayerNormV2GetWorkspaceSize'];assert all(any(l.split()[-1]==name for l in exports.splitlines()) for name in wanted)
b=R/'build-device-v1/binary/ascend950/bin';target=dest/'op_impl/ai_core/tbe/kernel';assert not target.exists();target.mkdir(parents=True);shutil.copytree(b/'ascend950',target/'ascend950');shutil.copytree(b/'config',target/'config');cfg=json.loads((target/'config/ascend950/binary_info_config.json').read_text());assert set(cfg)=={'AdaLayerNormV2'}
for item in cfg['AdaLayerNormV2']['binaryList']:
 for k in ['binPath','jsonPath']:assert (target/item[k]).is_file()
objects={str(p.relative_to(target)):sha(p) for p in target.rglob('*.o')};assert objects
files={str(p.relative_to(dest)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in dest.rglob('*') if p.is_file()};source_records={}
for rel in ['norm/ada_layer_norm_v2/op_host/ada_layer_norm_v2_def.cpp','norm/ada_layer_norm_v2/docs/aclnnAdaLayerNormV2.md']:
 source_records[rel]={'path':str(R/rel),'sha256':sha(R/rel)}
out={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_pin':manifest['pin'],'published_patch_sha256':manifest['patch_sha256'],'source_manifest_sha256':sha(N/'SOURCE_MANIFEST.json'),'source_tracked_files_verified':len(manifest['files']),'source_changes':[],'staging_root':str(dest),'libraries':{'op_api/lib/libcust_opapi.so':files['op_api/lib/libcust_opapi.so']},'selected_API_exports':wanted,'API_headers':{p.name:p.read_text() for p in (dest/'op_api/include').glob('*ada*')},'all_files':files,'device_objects':objects,'aggregate_config_sha256':sha(target/'config/ascend950/binary_info_config.json'),'selected_original_sources':source_records,'private_original_device_kernels_verified':False,'not_native_or_performance_qualification':True,'assembly_source_sha256':sha(__file__)};(N/'ADALNV2_NATIVE_PACKAGE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'source_files':len(manifest['files']),'artifact_files':len(files),'device_objects':len(objects),'manifest_sha256':sha(N/'ADALNV2_NATIVE_PACKAGE.json')}),flush=True)
