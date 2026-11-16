from pathlib import Path
import json,hashlib,shutil,datetime
B=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native');old=B/'ops-transformer-gmmv5-original-v1';N=B/'ops-transformer-gmmv5-aot-original-v2';m=json.loads((old/'SOURCE_MANIFEST.json').read_text());assert len(m['files'])==15360 and m['pin']=='7f642b905f1c095051fa3669063ddd708b40cb92';assert not N.exists();N.mkdir();(N/'source').mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,v in m['files'].items():
 p=old/'source'/n;assert sha(p)==v['sha256'];d=N/'source'/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d);assert sha(d)==v['sha256']
(N/'SOURCE_MANIFEST.json').write_text(json.dumps(m,indent=2)+chr(10));print(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':15360,'pin':m['pin'],'all_source_SHA_unchanged':True,'old_failure_and_JIT_package_preserved':True,'no_NPU_import_or_calls':True}))
