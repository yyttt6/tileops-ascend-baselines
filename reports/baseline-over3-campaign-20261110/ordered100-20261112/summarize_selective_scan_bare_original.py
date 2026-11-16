from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summarize():
 audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());rows=[]
 for cp in P.glob('rollout-torch-selective-scan-bare-*.CHECKPOINT.json'):
  e=json.loads(cp.read_text());a=audit[cp.name.removesuffix('.CHECKPOINT.json')];assert a['guard_valid'] and a['checkpoint_sha256']==sha(cp);root=P/'remote/runs'/e['tag'];assert sha(root/'SOURCE.json')==e['batch']['source_sha256'];source=json.loads((root/'SOURCE.json').read_text());assert any(k.endswith('/worker_selective_scan_bare_original.py') and v==sha(P/'worker_selective_scan_bare_original.py') for k,v in source.items())
  for r in e['batch']['results']:
   if r['status']!='raw_checked_pending_guard':continue
   assert r['ordinal']==1499 and r['key']==['torch-eager','SelectiveScanFwdOp','bare-scan-bfloat16'];assert next(x for x in a['rows'] if x['ordinal']==1499)['status']=='formal_raw_and_full_output_verified';proof=json.loads((root/'1499/FROZEN_SCAN_POLICY.json').read_text());assert proof['frozen_oracle_and_verification_FunctionDef_AST_unmodified'] and proof['original_normalized_bound']==1e-3 and proof['published_program_function_AST_unchanged'] and proof['both_complete_outputs_retained'];assert source[proof['frozen_source']]==proof['frozen_source_sha256'] and source[proof['frozen_numerics_source']]==proof['frozen_numerics_source_sha256'];rows.append({'ordinal':1499,'key':r['key'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'source_sha256':e['batch']['source_sha256'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'protocol':r['timing']['measurement_protocol'],'payload_sha256':r['payload_sha256'],'prepare_sha256':r['prepare_sha256'],'full_checker_15raw_guard_valid':True,'frozen_FP64_oracle_normalized_checker_unchanged':True,'both_outputs_checked':True,'exact_catalog_restored_not_alias':True,'no_historical_performance_substitution':True,'not_finalratio_or_paired_speedup':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_original_Torch_programs':len(rows),'rows':rows,'staged_not_published':True}
 for f in [P/'SELECTIVE_SCAN_BARE_ORIGINAL_CURRENT.json',P/'RESOLVED_SELECTIVE_SCAN_BARE_ORIGINAL.json',P.parent/'delivery/ordered_selective_scan_bare_original/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(f)
 return out
