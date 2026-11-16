"""Qualify two never-executed original Torch cases after earlier mixed CPUprepare failed."""
import pathlib,json,hashlib,datetime
P=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize():
 rows=[];label='rollout-tilelang-guard-recovery-1587-1590-v1';f=P/(label+'.CHECKPOINT.json')
 if f.exists():
  e=json.loads(f.read_text());audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text())[label];assert audit['checkpoint_sha256']==sha(f) and audit['guard_valid'];assert e['batch']['runtime']['torch']=='2.13.0+cpu'
  source=P/'remote/runs'/e['tag']/'SOURCE.json';assert sha(source)==e['batch']['source_sha256'];source_map=json.loads(source.read_text());worker=[v for k,v in source_map.items() if k.endswith('/worker_tilelang_guard_recovery.py')];assert worker==[sha(P/'worker_tilelang_guard_recovery.py')]
  for r in e['batch']['results']:
   assert r['ordinal'] in (1587,1588,1589,1590) and r['key'][0:2]==['tilelang','RMSNormFwdOp']
   if r['status']!='raw_checked_pending_guard':continue
   a=[x for x in audit['rows'] if x['ordinal']==r['ordinal']];assert len(a)==1 and a[0]['status']=='formal_raw_and_full_output_verified';assert r['timing']['measurement_protocol']=='ascend950-isolated-ffts-v3'
   rows.append({'ordinal':r['ordinal'],'key':r['key'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'api':r['api'],'source_sha256':e['batch']['source_sha256'],'checkpoint':f.name,'checkpoint_sha256':sha(f),'runtime':e['batch']['runtime'],'protocol':r['timing']['measurement_protocol'],'prepare_sha256':r['prepare_sha256'],'payload_sha256':r['payload_sha256'],'reference_policy':r['reference_policy'],'full_checker_15raw_guard_valid':True,'not_final_under3_ratio_closure':True,'no_alternate_or_speedup_claim':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_original_APIs':len(rows),'rows':rows,'no_historical_public_ratio_updated':True,'old_CPU_prepare_failure_retained':True}
 for f in [P/'TILELANG_GUARD_RECOVERY_CURRENT.json',P/'RESOLVED_TILELANG_GUARD_RECOVERY.json',P.parent/'delivery/ordered_tilelang_guard_recovery/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);t=f.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2));t.replace(f)
 return out
