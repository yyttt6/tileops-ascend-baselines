import pathlib,json,hashlib,datetime
P=pathlib.Path(__file__).resolve().parent
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def summarize():
 audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());rows=[];seen=set()
 for f in P.glob('rollout-nn-graph-boolall-*.CHECKPOINT.json'):
  label=f.name.removesuffix('.CHECKPOINT.json');e=json.loads(f.read_text());a=audit[label];assert a['checkpoint_sha256']==sha(f) and a['guard_valid'];assert e['batch']['runtime']['torch']=='2.13.0+cpu';source=P/'remote/runs'/e['tag']/'SOURCE.json';assert sha(source)==e['batch']['source_sha256'];m=json.loads(source.read_text());worker='worker_nn_graph_boolall.py' if label=='rollout-nn-graph-boolall-408-v2' else 'worker_nn_graph_boolall_expand.py';assert any(k.endswith('/'+worker) and v==sha(P/worker) for k,v in m.items())
  for r in e['batch']['results']:
   i=r['ordinal'];assert i in {406,407,408,409};assert r['key'][:2]==['ops-nn','AllFwdOp']
   if r['status']!='raw_checked_pending_guard':continue
   assert i not in seen,'Duplicate qualified result';seen.add(i)
   checked=[z for z in a['rows'] if z['ordinal']==i];assert len(checked)==1 and checked[0]['status']=='formal_raw_and_full_output_verified';assert r['timing']['measurement_protocol']=='ascend950-isolated-ffts-v3'
   rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'runtime':e['batch']['runtime'],'protocol':r['timing']['measurement_protocol'],'checkpoint':f.name,'checkpoint_sha256':sha(f),'source_sha256':e['batch']['source_sha256'],'selected_sources':r['sources'],'prepare_sha256':r['prepare_sha256'],'payload_sha256':r['payload_sha256'],'full_checker_15raw_guard_valid':True,'not_final_under3_ratio_closure':True,'no_alternate_or_speedup_claim':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_original_APIs':len(rows),'rows':rows,'no_historical_public_ratio_updated':True,'old_dependency_deferrals_retained':True}
 for f in [P/'NN_GRAPH_BOOLALL_CURRENT.json',P/'RESOLVED_NN_GRAPH_BOOLALL.json',P.parent/'delivery/ordered_nn_graph_boolall/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);t=f.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2));t.replace(f)
 return out
