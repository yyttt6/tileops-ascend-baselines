"""Read-only exact-scope publication-field audit; not a performance acceptance."""
from pathlib import Path
import json,hashlib,collections
P=Path(__file__).resolve().parent
source=P.parent/'baseline-speedup-audit-20261108/EVIDENCE_INDEX.json'
index=json.loads(source.read_text());queue=json.loads((P/'QUEUE.json').read_text());rows=[]
for q in queue:
 mid=q.get('measurement');entry=index.get(mid)
 r={'key':[q['library'],q['op'],q['case']],'measurement_id':mid,'published_us':float(q['baseline_us'])}
 if not entry:r['status']='source_not_indexed'
 else:
  e=entry.get('evidence',{});assert e.get('key')==r['key'], 'measurement identity mismatch';t=e.get('timing',{});device=t.get('latency_ms');host=e.get('end_to_end_mean_ms');metric=e.get('comparison_metric');mode=e.get('timing_mode')
  r.update(source_kind=entry.get('kind'),declared_metric=metric,timing_mode=mode,source_device_us=None if device is None else 1000*device,source_host_us=None if host is None else 1000*host)
  matchdevice=device is not None and abs(r['published_us']-1000*device)<1e-6
  matchhost=host is not None and abs(r['published_us']-1000*host)<1e-6
  r.update(published_matches_device=matchdevice,published_matches_host=matchhost)
  if metric=='end_to_end_mean_ms' and matchdevice and not matchhost:r['status']='declared_host_published_device'
  elif metric=='end_to_end_mean_ms' and matchhost:r['status']='declared_host_matches'
  elif metric=='device_latency_ms' and matchdevice:r['status']='declared_device_matches'
  else:r['status']='unresolved_metric_or_value'
 rows.append(r)
assert len(rows)==2208 and len({tuple(r['key']) for r in rows})==2208
counts=collections.Counter(r['status'] for r in rows);bylib=collections.Counter(r['key'][0] for r in rows if r['status']=='declared_host_published_device')
result={'scope':'frozen2208 exact pairs only','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'counts':dict(counts),'declared_host_published_device_by_library':dict(bylib),'caveat':'Field/provenance discrepancy, not proof which metric is intended by current site. No baseline value, route, cache, queue review status or kernel modified. Never form host/device mixed ratio; both sides require same metric.','rows':rows}
(P/'PUBLICATION_METRIC_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'counts':dict(counts),'by_library':dict(bylib)},indent=2))
