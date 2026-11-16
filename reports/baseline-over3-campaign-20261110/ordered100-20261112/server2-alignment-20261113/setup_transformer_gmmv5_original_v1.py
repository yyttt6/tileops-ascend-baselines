from pathlib import Path
import json,hashlib,shutil,datetime
B=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native');old=B/'ops-transformer-arithmetic-v1';N=B/'ops-transformer-gmmv5-original-v1';m=json.loads((B/'ops-transformer-selection-v1/SOURCE_MANIFEST.json').read_text());assert m['pin']=='7f642b905f1c095051fa3669063ddd708b40cb92' and len(m['files'])==15360;assert not N.exists();N.mkdir();(N/'source').mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,v in m['files'].items():
 src=old/'source'/n;assert sha(src)==v['sha256'];dst=N/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(dst)==v['sha256']
(N/'SOURCE_MANIFEST.json').write_text(json.dumps(m,indent=2)+chr(10));print(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':len(m['files']),'pin':m['pin'],'original_all_source_SHA_verified':True,'no_NPU_import_or_shared_source_changes':True}))
