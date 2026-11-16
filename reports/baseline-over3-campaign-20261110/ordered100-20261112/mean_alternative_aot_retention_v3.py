from pathlib import Path
import json,hashlib,shutil,datetime
N=Path("/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-mean-original-v1");sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def capture(work,destination):
 out=Path(destination);assert not out.exists();out.mkdir();m=json.loads((N/'MEAN_ORIGINAL_PACKAGE_V3.json').read_text());vendor=Path(m['vendor']);maps=Path('/proc/self/maps').read_text();assert str(vendor/'op_api/lib/libcust_opapi.so') in maps
 for name,r in m['files'].items():assert sha(vendor/name)==r['sha256']
 shutil.copytree(vendor,out/'PACKAGE');shutil.copy2(N/'MEAN_ORIGINAL_PACKAGE_V3.json',out/'PACKAGE_MANIFEST.json');shutil.copy2(N/'SOURCE_MANIFEST.json',out/'SOURCE_MANIFEST.json');records=[]
 for p in (vendor/'op_impl/ai_core/tbe/kernel/ascend950/reduce_mean_apt').glob('*.json'):
  d=json.loads(p.read_text());obj=p.with_name(d['binFileName']+d['binFileSuffix']);assert sha(obj)==d['sha256'];records.append({'metadata_relative':str(p.relative_to(vendor)),'metadata':d,'device_ELF_sha256':sha(obj)})
 assert len(records)==m['device_objects'] and records;r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,'loaded_host_path':str(vendor/'op_api/lib/libcust_opapi.so'),'loaded_host_sha256':sha(vendor/'op_api/lib/libcust_opapi.so'),'package_manifest_sha256':sha(N/'MEAN_ORIGINAL_PACKAGE_V3.json'),'source_manifest_sha256':sha(N/'SOURCE_MANIFEST.json'),'outside_timed_windows':True};(out/'RETENTION.json').write_text(json.dumps(r,indent=2));(out/'loaded-maps.txt').write_text(maps);return r
