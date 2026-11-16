"""Current exact original PTO API qualification; immutable old receipts not reused."""
from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summarize():
 scope={x['ordinal'] for x in json.loads((P/'PTO_FROZEN_KEY_READINESS_CPU.json').read_text())['frozen_unresolved_key_candidates']};audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());rows=[];seen=set()
 for f in sorted(P.glob('rollout-pto-current-*.CHECKPOINT.json')):
  e=json.loads(f.read_text());label=f.name.removesuffix('.CHECKPOINT.json');a=audit[label];assert a['checkpoint_sha256']==sha(f) and a['guard_valid'];assert e['guard']['owner']=='root/baseline-ordered100/'+e['tag'];assert e['batch']['runtime']['torch']=='2.13.0+cpu';source=P/'remote/runs'/e['tag']/'SOURCE.json';assert sha(source)==e['batch']['source_sha256'];sm=json.loads(source.read_text())
  if not any(r['status']=='raw_checked_pending_guard' for r in e['batch']['results']):continue  # retain failed attempts; never count them or impose new helper bytes on old failed source
  assert [v for k,v in sm.items() if k.endswith('/worker_pto_current.py')]==[sha(P/'worker_pto_current.py')];assert [v for k,v in sm.items() if k.endswith('/pto_current_binding.py')]==[sha(P/'pto_current_binding.py')]
  for r in e['batch']['results']:
   i=r['ordinal'];assert i in scope and i not in seen;seen.add(i)
   if r['status']!='raw_checked_pending_guard':continue
   checks=[x for x in a['rows'] if x['ordinal']==i];assert len(checks)==1 and checks[0]['status']=='formal_raw_and_full_output_verified';assert r['timing']['measurement_protocol']=='ascend950-isolated-ffts-v3'
   proof=json.loads((P/'remote/runs'/e['tag']/str(i)/'PTO_CURRENT_BINDING.json').read_text());assert proof['actual_owner']==e['guard']['owner'] and proof['full_reverse_guard_AST_identical'] and proof['full_original_contract_identical'] and proof['pins_verified']==864
   jit=P/'remote/runs'/e['tag']/str(i)/'PTO_JIT_ARTIFACTS.json';assert json.loads(jit.read_text()),'Generated native artifact evidence required'
   rows.append({'ordinal':i,'key':r['key'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'api':r['api'],'source_sha256':e['batch']['source_sha256'],'checkpoint':f.name,'checkpoint_sha256':sha(f),'runtime':e['batch']['runtime'],'protocol':r['timing']['measurement_protocol'],'prepare_sha256':r['prepare_sha256'],'payload_sha256':r['payload_sha256'],'reference_policy':r['reference_policy'],'full_checker_15raw_guard_valid':True,'not_final_under3_ratio_closure':True,'no_alternate_or_speedup_claim':True,'guard_binding':proof,'jit_artifacts_sha256':sha(jit)})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_original_APIs':len(rows),'rows':rows,'no_public_or_historical_ratio_update':True,'old_owner_not_spoofed':True,'kernel_compiler_precision_unchanged':True}
 for f in [P/'PTO_CURRENT.json',P/'RESOLVED_PTO_CURRENT.json',P.parent/'delivery/ordered_pto_current/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);t=f.with_suffix('.summary.tmp');t.write_text(json.dumps(out,indent=2)+'\n');t.replace(f)
 return out
