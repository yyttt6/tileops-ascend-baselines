"""CPU staging of source-path/cache repairs; full original APIs really executed."""
from pathlib import Path
import json,hashlib,shutil,fcntl,datetime
P=Path(__file__).resolve().parent;D=P.parent/'delivery/ordered_source_path_repairs';D.mkdir(exist_ok=True)
with (P/'audit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text())
manifest=json.loads((P/'RESOLVED_SOURCE_PATHS.json').read_text());routes=[]
for i,row in manifest['rows'].items():
 cp=P/row['accepted_checkpoint'];digest=hashlib.sha256(cp.read_bytes()).hexdigest();assert digest==row['accepted_checkpoint_sha256'];e=json.loads(cp.read_text());assert audit[e['label']]['checkpoint_sha256']==digest;assert e['guard']['performance_valid'] and e['exit_code']==0
 r=e['batch']['results'][0];assert r['status']=='raw_checked_pending_guard' and r['ordinal']==int(i);assert r['check']['checked_outputs']==r['check']['total_outputs']>0 and not r['check']['unchecked_reason'];assert len(r['timing']['latency_samples_ms'])==15
 if row['key'][0]=='vllm-ascend':
  helper=P/'workspace/isolated_wrappers/vllm_external_source_hashes.py';expected=[h for path,h in r['sources'].items() if path.endswith('/isolated_wrappers/vllm_external_source_hashes.py')];assert len(expected)==1 and hashlib.sha256(helper.read_bytes()).hexdigest()==expected[0];shutil.copyfile(helper,D/helper.name)
 routes.append(row)
f=D/'ROUTES.json'
if f.exists():f.with_name(f.name+'.before-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).write_bytes(f.read_bytes())
tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(routes,indent=2));tmp.replace(f)
(D/'DELIVERY.txt').write_text('Exact accepted original API deployment repairs only. External original extension paths retain actual absolute path and full SHA. FP8 cache revision changes only the audited source metadata path-check identity, preserving inputs/descriptors. All measured original APIs really executed, with full checker,15raw replay and valid whole-process guard. Original kernel/timing/precision unchanged. Original failures retained; no historical/public speedup claim.\n')
print('STAGED',len(routes),'original API repairs')
