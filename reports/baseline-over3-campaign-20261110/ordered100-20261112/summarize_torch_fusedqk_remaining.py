"""Finite remaining original Torch math; full checks/raw/source/guard required."""
from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summarize():
 rows=[];seen=set();audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());proof=json.loads((P/'TORCH_FUSEDQK_REMAINING_CPU_PROOF.json').read_text());assert proof['CPU_cases']==24 and proof['proof']['full_reverse_AST_identical']
 for f in sorted(P.glob('rollout-torch-fusedqk-remaining-*.CHECKPOINT.json')):
  e=json.loads(f.read_text());label=f.name.removesuffix('.CHECKPOINT.json');a=audit[label];assert a['checkpoint_sha256']==sha(f) and a['guard_valid'];assert e['batch']['runtime']['torch']=='2.13.0+cpu'
  source=P/'remote/runs'/e['tag']/'SOURCE.json';assert sha(source)==e['batch']['source_sha256'];sm=json.loads(source.read_text());assert [v for k,v in sm.items() if k.endswith('/worker_torch_fusedqk_remaining.py')]==[sha(P/'worker_torch_fusedqk_remaining.py')];assert [v for k,v in sm.items() if k.endswith('/torch_fusedqk_remaining_reference.py')]==[proof['helper_sha256']];assert [v for k,v in sm.items() if k.endswith('/workloads/ascend_extended.py')]==[proof['original_workload_sha256']]
  for r in e['batch']['results']:
   i=r['ordinal'];assert i in range(1356,1361) and i not in seen;seen.add(i)
   if r['status']!='raw_checked_pending_guard':continue
   checks=[x for x in a['rows'] if x['ordinal']==i];assert len(checks)==1 and checks[0]['status']=='formal_raw_and_full_output_verified';assert r['timing']['measurement_protocol']=='ascend950-isolated-ffts-v3'
   rows.append({'ordinal':i,'key':r['key'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'api':r['api'],'source_sha256':e['batch']['source_sha256'],'checkpoint':f.name,'checkpoint_sha256':sha(f),'runtime':e['batch']['runtime'],'protocol':r['timing']['measurement_protocol'],'prepare_sha256':r['prepare_sha256'],'payload_sha256':r['payload_sha256'],'reference_policy':r['reference_policy'],'full_checker_15raw_guard_valid':True,'not_final_under3_ratio_closure':True,'no_alternate_or_speedup_claim':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_original_APIs':len(rows),'rows':rows,'no_historical_public_ratio_updated':True,'original_CPU_oracle_unchanged':True}
 for f in [P/'TORCH_FUSEDQK_REMAINING_CURRENT.json',P/'RESOLVED_TORCH_FUSEDQK_REMAINING.json',P.parent/'delivery/ordered_torch_fusedqk_remaining/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);t=f.with_suffix('.summary.tmp');t.write_text(json.dumps(out,indent=2)+'\n');t.replace(f)
 return out
