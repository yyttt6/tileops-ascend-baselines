from pathlib import Path
import json,hashlib,shutil,datetime
B=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native');old=B/'fla-original-chunk-v1';new=B/'fla-original-gate-v1';manifest=json.loads((old/'FLA_ORIGINAL_COMPLETE_SOURCE_V3.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert len(manifest['files'])==5479
assert not new.exists(),'Preserve prior state and diagnose instead of overwrite';new.mkdir();(new/'source').mkdir()
for name,h in manifest['files'].items():
 src=old/'source'/name;assert sha(src)==h;dst=new/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(dst)==h
(new/'ORIGINAL_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+chr(10));print(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':5479,'source_manifest_sha256':sha(new/'ORIGINAL_SOURCE_MANIFEST.json'),'private_source':str(new/'source'),'no_NPU_import_or_shared_source_change':True}))
