"""CPU-only assembly from original CPack staging, no installed SDK mutations."""
from pathlib import Path
import json,hashlib,shutil,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-attention-v1');m=json.loads((N.with_name('ops-transformer-selection-v1')/'SOURCE_MANIFEST.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,r in m['files'].items():assert sha(N/'source'/name)==r['sha256'],name
roots=[p for p in (N/'source/build/_CPack_Packages').glob('**/attentionpairs_transformer') if p.is_dir()];assert len(roots)==1
src=roots[0];dst=N/'private-v1/vendors/attentionpairs_transformer';assert not dst.exists();dst.parent.mkdir(parents=True);shutil.copytree(src,dst,symlinks=True)
files={}
for p in dst.rglob('*'):
 if p.is_file():files[str(p.relative_to(dst))]={'sha256':sha(p),'bytes':p.stat().st_size}
assert any('flash_attention_score' in x for x in files)
package=next((N/'source/build').glob('*.run'));r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'original_host_and_JIT_source_package_assembled_not_native_qualified','source_pin':m['pin'],'source_tracked_files_unchanged':len(m['files']),'vendor':str(dst),'CPack_staging':str(src),'package':str(package),'package_sha256':sha(package),'files':files,'kernel_or_generated_flags_edits':False,'no_shared_OPP_changes':True};(N/'TRANSFORMER_ATTENTION_PACKAGE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'files':len(files),'vendor':str(dst),'package_sha256':r['package_sha256']}))
