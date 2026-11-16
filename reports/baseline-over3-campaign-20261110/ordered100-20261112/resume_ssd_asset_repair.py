"""CPU bookkeeping after a safe controller boundary; preserve all former failures."""
from pathlib import Path
import json,datetime,hashlib
P=Path(__file__).resolve().parent
assert not Path('/proc/914840').exists(),'old foreground controller still active'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
f=P/'ROLLOUT_LEDGER.json';raw=f.read_bytes();f.with_name(f.name+'.before-ssdasset-'+stamp).write_bytes(raw);ledger=json.loads(raw);labels=set();removed=[]
for i in [1080,1081,1082]:
 e=ledger[str(i)];r=e['result'];assert r['status'] in ('failed','provider_dependency_deferred')
 if r['status']=='failed':
  assert r['phase']=='factory' and 'causal_256.json' in r['traceback'];labels.add(e['label'])
 removed.append({'ordinal':i,'entry':e});del ledger[str(i)]
f=P/'PINNED_DEPENDENCY_SUPERSEDED.json';raw=f.read_bytes();f.with_name(f.name+'.before-ssdasset-'+stamp).write_bytes(raw);j=json.loads(raw);j['checkpoint_labels']=sorted(set(j['checkpoint_labels'])|labels);f.write_text(json.dumps(j,indent=2));f=P/'ROLLOUT_LEDGER.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(f)
f=P/'SGL_SSD_LAYOUT_ASSET_REPAIR.json';j=json.loads(f.read_text());j.update(previous_failed_or_deferred_entries=removed,requeued_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='original asset restored and SHA verified; three original cases requeued with new label, not yet accepted');f.write_text(json.dumps(j,indent=2));print('requeued three SSD cases; original/candidate checkpoints stay separate',flush=True)
