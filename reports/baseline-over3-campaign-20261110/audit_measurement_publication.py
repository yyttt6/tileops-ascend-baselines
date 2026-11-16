"""CPU-only field audit of measurement-schema records; not performance acceptance."""
from pathlib import Path
import json,hashlib,gzip,collections
P=Path(__file__).resolve().parent
idx=json.loads((P.parent/'baseline-speedup-audit-20261108/EVIDENCE_INDEX.json').read_text())
prior=json.loads((P/'PUBLICATION_METRIC_AUDIT.json').read_text());rows=[]
for row in prior['rows']:
 if row.get('source_kind')!='measurement':continue
 r=dict(row);entry=idx[r['measurement_id']];base=Path(entry['directory']);f=base/'measurement.json'
 if not f.exists():f=base/'measurement.json.gz'
 if not f.exists():r['followup_status']='measurement_file_missing';rows.append(r);continue
 raw=gzip.decompress(f.read_bytes()) if f.suffix=='.gz' else f.read_bytes();m=json.loads(raw)
 assert m['key']==r['key'],f
 r.update(measurement_file=str(f),measurement_json_sha256=hashlib.sha256(raw).hexdigest(),explicit_metric=m.get('comparison_metric'),explicit_mode=m.get('timing_mode'))
 t=m.get('timing',{});value=t.get('latency_ms');samples=t.get('latency_samples_ms',[])
 r.update(measurement_device_us=None if value is None else value*1000,sample_count=len(samples),profile_count=len(t.get('retained_profiles',[])))
 r['followup_status']='published_matches_measurement_device_metric_unspecified' if value is not None and abs(value*1000-r['published_us'])<1e-6 else 'measurement_value_unresolved'
 rows.append(r)
assert len(rows)==561
out={'scope':'561 measurement-schema records within frozen2208','counts':dict(collections.Counter(r['followup_status'] for r in rows)),'caveat':'Source JSON key checked and SHA recorded; field comparison only, no archive replay or new performance acceptance. Missing explicit comparison_metric is not inferred as a declared metric.','rows':rows}
(P/'MEASUREMENT_PUBLICATION_FIELD_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['counts'])
