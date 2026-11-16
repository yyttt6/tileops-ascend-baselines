from pathlib import Path
import json,hashlib,shutil,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-gate-v1');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def capture(work,destination):
 out=Path(destination);out.mkdir(parents=True,exist_ok=False);manifest=N/'PACKAGE_AOT_V1_MANIFEST.json';m=json.loads(manifest.read_text());package=Path(m['package']);identity=json.loads((N/'ORIGINAL_SOURCE_MANIFEST.json').read_text());assert all(sha(N/'source'/n)==h for n,h in identity['files'].items());files={}
 def keep(p):
  key=str(p.relative_to(N));dst=out/key;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);files[key]={'sha256':sha(dst),'bytes':dst.stat().st_size};return key
 for name,h in m['artifact_sha256'].items():
  p=package/name;assert sha(p)==h;keep(p)
 for p in [manifest,N/'ORIGINAL_SOURCE_MANIFEST.json',N/'stage_fla_gate_aot_v1.py',N/'build-host-fla-gate-v1.sh']:keep(p)
 records=[]
 for meta in sorted((package/'opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/kernel/ascend950/kda_gate_cumsum').glob('*.json')):
  data=json.loads(meta.read_text());obj=meta.with_name(data['binFileName']+data['binFileSuffix']);built=N/'source/build/binary/ascend950/bin/kda_gate_cumsum'/obj.name;assert sha(obj)==data['sha256']==sha(built);keep(built)
  original=N/'source/fla/ops/ascendc/kda/kda_gate_cumsum/op_kernel/kda_gate_cumsum.cpp';compiled=N/'source/build/binary/ascend950/src/kda_gate_cumsum/op_kernel/kda_gate_cumsum.cpp';assert sha(original)==sha(compiled)==identity['files'][str(original.relative_to(N/'source'))];keep(original);keep(compiled)
  records.append({'metadata':data,'metadata_path':str(meta.relative_to(N)),'object_path':str(obj.relative_to(N)),'build_object_path':str(built.relative_to(N)),'original_CPP_path':str(original.relative_to(N)),'compiled_CPP_path':str(compiled.relative_to(N)),'original_CPP_sha256':sha(original)})
 assert records;maps=Path('/proc/self/maps').read_text();library=package/'opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so';assert str(library) in maps;(out/'loaded-maps.txt').write_text(maps)
 r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'outside_timed_windows':True,'original_AoT_source_build_package_identity_verified':True,'files':files,'records':records,'loaded_library':str(library),'loaded_library_sha256':sha(library),'source_manifest_sha256':sha(N/'ORIGINAL_SOURCE_MANIFEST.json'),'no_source_precision_checker_changes':True};(out/'RETENTION.json').write_text(json.dumps(r,indent=2)+chr(10));return r
