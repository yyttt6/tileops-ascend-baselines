from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summarize():
 audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());proof=json.loads((P/'SELECTIVE_SCAN_ORIGINAL_CPU_DIAGNOSIS.json').read_text());rows=[]
 for cp in P.glob('rollout-torch-selective-scan-original-*.CHECKPOINT.json'):
  e=json.loads(cp.read_text());a=audit[cp.name.removesuffix('.CHECKPOINT.json')];assert a['guard_valid'] and a['checkpoint_sha256']==sha(cp)
  source=P/'remote/runs'/e['tag']/'SOURCE.json';assert sha(source)==e['batch']['source_sha256'];sources=json.loads(source.read_text());assert any(k.endswith('/'+name) and v==sha(P/name) for name in ['worker_selective_scan_original_device.py','worker_selective_scan_original_device_v2.py','worker_selective_scan_original_expansion.py'] for k,v in sources.items())
  for r in e['batch']['results']:
   if r['status']!='raw_checked_pending_guard':continue
   i=r['ordinal'];assert i in {1497,1498,1500,1501,1502};meta_file=P/'remote/prepared-rollout'/str(i)/'prepare.json';meta=json.loads(meta_file.read_text());assert sha(meta_file)==r['prepare_sha256'] and meta['key']==r['key'] and meta['payload_sha256']==r['payload_sha256'];assert r['reference_policy']=='original_npu';assert next(x for x in a['rows'] if x['ordinal']==i)['status']=='formal_raw_and_full_output_verified';boundary=json.loads((source.parent/str(i)/'TORCH_DEVICE_BOUNDARY.json').read_text());assert boundary['existing_frozen_CPU_FP64_oracle_unchanged'] and boundary['published_function_AST_unchanged'] and boundary['original_published_FP32_math_not_precision_edited'] and boundary['both_y_and_final_states_retained'];assert sources[proof['original_oracle_source']]==proof['original_oracle_source_sha256'] and sources[proof['published_original_program_source']]==proof['published_original_program_source_sha256'];rows.append({'ordinal':i,'key':r['key'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'source_sha256':e['batch']['source_sha256'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'api':r['api'],'protocol':r['timing']['measurement_protocol'],'payload_sha256':r['payload_sha256'],'prepare_sha256':r['prepare_sha256'],'full_checker_15raw_guard_valid':True,'original_frozen_oracle_checker_payload_unchanged':True,'published_program_AST_unchanged':True,'no_invalid_CPUtiming_used_as_device_control':True,'not_final_ratio_or_paired_speedup':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_original_Torch_programs':len(rows),'rows':rows,'staged_not_published':True}
 for f in [P/'SELECTIVE_SCAN_ORIGINAL_CURRENT.json',P/'RESOLVED_SELECTIVE_SCAN_ORIGINAL.json',P.parent/'delivery/ordered_selective_scan_original/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(f)
 return out
