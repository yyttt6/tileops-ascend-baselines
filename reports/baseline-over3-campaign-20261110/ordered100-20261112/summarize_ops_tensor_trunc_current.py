from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summarize():
 label='rollout-ops-tensor-trunc-current-735-v1';f=P/(label+'.CHECKPOINT.json');rows=[]
 if f.exists():
  e=json.loads(f.read_text());a=json.loads((P/'ROLLOUT_AUDIT.json').read_text())[label];assert a['checkpoint_sha256']==sha(f) and a['guard_valid'];assert e['batch']['runtime']['torch']=='2.13.0+cpu';s=P/'remote/runs'/e['tag']/'SOURCE.json';assert sha(s)==e['batch']['source_sha256'];sm=json.loads(s.read_text());assert [v for k,v in sm.items() if k.endswith('/worker_ops_tensor_trunc_current.py')]==[sha(P/'worker_ops_tensor_trunc_current.py')]
  for r in e['batch']['results']:
   assert r['ordinal']==735 and r['key']==['ops-tensor','TruncFwdOp','16M-float32'] and r['status']=='raw_checked_pending_guard';checks=[x for x in a['rows'] if x['ordinal']==735];assert len(checks)==1 and checks[0]['status']=='formal_raw_and_full_output_verified';assert r['timing']['measurement_protocol']=='ascend950-isolated-ffts-v3'
   rows.append({'ordinal':735,'key':r['key'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'api':r['api'],'source_sha256':e['batch']['source_sha256'],'checkpoint':f.name,'checkpoint_sha256':sha(f),'runtime':e['batch']['runtime'],'protocol':r['timing']['measurement_protocol'],'prepare_sha256':r['prepare_sha256'],'payload_sha256':r['payload_sha256'],'reference_policy':r['reference_policy'],'full_checker_15raw_guard_valid':True,'not_final_under3_ratio_closure':True,'no_alternate_or_speedup_claim':True,'selected_sources':r['sources']})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_original_APIs':len(rows),'rows':rows,'old256Mtimeout_not_retried':True,'no_public_ratio_updated':True}
 for f in [P/'OPS_TENSOR_TRUNC_CURRENT.json',P/'RESOLVED_OPS_TENSOR_TRUNC_CURRENT.json',P.parent/'delivery/ordered_ops_tensor_trunc_current/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);t=f.with_suffix('.summary.tmp');t.write_text(json.dumps(out,indent=2)+'\n');t.replace(f)
 return out
if __name__=='__main__':print('QUALIFIED',summarize()['accepted_original_APIs'])
