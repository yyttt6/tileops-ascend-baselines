from pathlib import Path
import json,hashlib,shutil,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def capture(destination):
 out=Path(destination);assert not out.exists();out.mkdir();m=json.loads((N/'GEMM_PAD4_NATIVE_PACKAGE.json').read_text());v=Path(m['staging_root']);maps=Path('/proc/self/maps').read_text();assert str(v/'op_api/lib/libcust_opapi.so') in maps
 for name,r in m['all_files'].items():assert sha(v/name)==r['sha256']
 source_manifest=Path(m['original_source_manifest']);assert sha(source_manifest)==m['source_manifest_sha256'];shutil.copy2(source_manifest,out/'SOURCE_MANIFEST.json');shutil.copytree(v,out/'PACKAGE');shutil.copy2(N/'GEMM_PAD4_NATIVE_PACKAGE.json',out/'PACKAGE_MANIFEST.json');shutil.copy2(N/'GEMM_PAD4_HOST_QUALIFICATION.json',out/'HOST_QUALIFICATION.json');records=[]
 for name,h in m['device_objects'].items():
  obj=v/'op_impl/ai_core/tbe/kernel'/name;md=json.loads(obj.with_suffix('.json').read_text());assert sha(obj)==h==md['sha256'];records.append({'metadata_relative':str(obj.with_suffix('.json').relative_to(v)),'metadata':md,'device_ELF_sha256':h})
 assert records;r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,'loaded_host_path':str(v/'op_api/lib/libcust_opapi.so'),'loaded_host_sha256':sha(v/'op_api/lib/libcust_opapi.so'),'package_manifest_sha256':sha(N/'GEMM_PAD4_NATIVE_PACKAGE.json'),'host_qualification_sha256':sha(N/'GEMM_PAD4_HOST_QUALIFICATION.json'),'source_manifest_sha256':sha(source_manifest),'outside_timed_windows':True};(out/'RETENTION.json').write_text(json.dumps(r,indent=2));(out/'loaded-maps.txt').write_text(maps)
