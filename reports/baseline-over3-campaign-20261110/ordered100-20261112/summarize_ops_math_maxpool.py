import pathlib,json,hashlib,datetime
P=pathlib.Path(__file__).resolve().parent
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def summarize():
 audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());rows=[];seen=set()
 for f in P.glob('rollout-ops-math-maxpool-*.CHECKPOINT.json'):
  label=f.name.removesuffix('.CHECKPOINT.json');e=json.loads(f.read_text());a=audit[label];assert a['checkpoint_sha256']==sha(f) and a['guard_valid'];assert e['batch']['runtime']['torch']=='2.13.0+cpu';source=P/'remote/runs'/e['tag']/'SOURCE.json';assert sha(source)==e['batch']['source_sha256'];m=json.loads(source.read_text());assert any(k.endswith('/worker_ops_math_maxpool.py') and v==sha(P/'worker_ops_math_maxpool.py') for k,v in m.items())
  for r in e['batch']['results']:
   i=r['ordinal'];assert i in {205, 206, 207, 336, 340, 341, 342, 352, 353} and i not in seen;seen.add(i);assert r['key'][0]=='ops-math' and r['key'][1] in {'AdaptiveMaxPool2dFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp'}
   if r['status']!='raw_checked_pending_guard':continue
   checked=[z for z in a['rows'] if z['ordinal']==i];assert len(checked)==1 and checked[0]['status']=='formal_raw_and_full_output_verified';assert r['timing']['measurement_protocol']=='ascend950-isolated-ffts-v3'
   rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'runtime':e['batch']['runtime'],'protocol':r['timing']['measurement_protocol'],'checkpoint':f.name,'checkpoint_sha256':sha(f),'source_sha256':e['batch']['source_sha256'],'selected_sources':r['sources'],'prepare_sha256':r['prepare_sha256'],'payload_sha256':r['payload_sha256'],'full_checker_15raw_guard_valid':True,'not_final_under3_ratio_closure':True,'not_paired_speedup_or_finalratio':True,'selected_complete_native_API':'complete original Amax; original full signed-window-packing run/reduce AST retained'})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_complete_APIs':len(rows),'rows':rows,'no_historical_public_ratio_updated':True,'old_dependency_deferrals_retained':True}
 for f in [P/'OPS_MATH_MAXPOOL_CURRENT.json',P/'RESOLVED_OPS_MATH_MAXPOOL.json',P.parent/'delivery/ordered_ops_math_maxpool/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);t=f.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2));t.replace(f)
 return out
