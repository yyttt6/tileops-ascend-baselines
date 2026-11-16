from pathlib import Path
import json,hashlib,shutil,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def capture(destination):
 out=Path(destination);assert not out.exists();out.mkdir();q=json.loads((N/'ORIGINAL_NORM_HOST_QUALIFICATION.json').read_text());assert q['full_API_embedded_kernel_source_build_guard_and_dual_archive_verified']
 maps=Path('/proc/self/maps').read_text();binary=N/'libops_math_layernorm.so';assert str(binary) in maps and sha(binary)==q['build_record']['binary_sha256'];assert sha(N/'layer_norm.asc')==q['original_source_sha256'];files={}
 for name in ['ORIGINAL_NORM_HOST_QUALIFICATION.json','BUILD_RECEIPT.json','libops_math_layernorm.so','layer_norm.asc']:
  src=N/name;shutil.copy2(src,out/name);assert sha(src)==sha(out/name);files[name]=sha(src)
 (out/'loaded-maps.txt').write_text(maps);(out/'RETENTION.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'loaded_host_path':str(binary),'loaded_host_sha256':sha(binary),'qualified_original_kernel_name':'_ZN10ascend_ops9LayerNorm15layer_norm_fp32EPhS1_S1_S1_llllf','no_native_recompile':True,'original_API_and_row_affine_AST_unchanged':True},indent=2))
