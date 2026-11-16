from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
scope=json.loads((P/'vllm-avgpool-source-review/REVIEW.json').read_text())['scope'];expected={tuple(x['key']) for x in scope};rows=[];seen=set();proof={}
for group in ['REGULAR','GLOBAL','OVERLAP','ADAPTIVE','CEIL']:
 f=P/f'VLLM_AVGPOOL_{group}_DUAL_METRIC.partial.json';d=json.loads(f.read_text());proof[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 for x in d['pairs']:
  matches=[k for k in expected if k[2]==x['case']];assert len(matches)==1;k=matches[0];assert k not in seen;seen.add(k)
  rows.append({'key':list(k),'group':group,'original_host_us':x['original_host_us'],'candidate_host_us':x['candidate_host_us'],'original_device_us':x['original_device_us'],'candidate_device_us':x['candidate_device_us'],'host_reduction_pct':100*x['host_reduction_fraction'],'device_change_pct':100*x['device_change_fraction'],'observed_lower_host':'fixed' if x['candidate_host_us']<x['original_host_us'] else 'native','observation_only':True,'evidence':f.name})
out={'scope':len(expected),'dual_pairs_verified':len(rows),'rows':rows,'pending':[list(k) for k in sorted(expected-seen)],'evidence_sha256':proof,'public_ratio_update':False,'warning':'Host-completed wrapper comparison and device readout must stay separate. Observed lower host is not cross-environment ratio acceptance or automatic installation.'}
(P/'VLLM_AVGPOOL_FAMILY_DUAL_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n');print(len(rows),'of',len(expected),'verified; pending',out['pending'])
