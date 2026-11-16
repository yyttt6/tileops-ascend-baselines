from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def summarize():
 audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());rows=[]
 for cp in P.glob('rollout-torch-gelu3-original-f037-*.CHECKPOINT.json'):
  e=json.loads(cp.read_text());a=audit[cp.name.removesuffix('.CHECKPOINT.json')];assert a['guard_valid'] and a['checkpoint_sha256']==sha(cp);src=P/'remote/runs'/e['tag']/'SOURCE.json';assert sha(src)==e['batch']['source_sha256'];sources=json.loads(src.read_text());assert any(k.endswith('/worker_torch_gelu3_original_f037.py') and v==sha(P/'worker_torch_gelu3_original_f037.py') for k,v in sources.items())
  for r in e['batch']['results']:
   if r['status']!='raw_checked_pending_guard':continue
   i=r['ordinal'];assert i in {1361,1362,1363} and r['measured_library']=='torch-eager' and r['source_commit']=='f037318f1483a8e4b17c30989e86debf0751a489' and r['measured_original_storage_Torch_program'] and r['reference_policy']=='original_npu';assert next(x for x in a['rows'] if x['ordinal']==i)['status']=='formal_raw_and_full_output_verified';rows.append({'ordinal':i,'key':r['key'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'api':r['api'],'runtime':e['batch']['runtime'],'protocol':r['timing']['measurement_protocol'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'source_sha256':e['batch']['source_sha256'],'payload_sha256':r['payload_sha256'],'prepare_sha256':r['prepare_sha256'],'prepare_metadata_path':r['prepare_metadata_path'],'full_checker_15raw_guard_valid':True,'not_paired_speedup_or_finalratio':True,'exact_original_catalog_recovered_not_alias':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_original_Torch_programs':len(rows),'rows':rows,'source_commit':'f037318f1483a8e4b17c30989e86debf0751a489','no_historical_or_public_ratio_update':True}
 for f in [P/'TORCH_GELU3_ORIGINAL_CURRENT.json',P/'RESOLVED_TORCH_GELU3_ORIGINAL.json',P.parent/'delivery/ordered_torch_gelu3_original/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(f)
 return out
