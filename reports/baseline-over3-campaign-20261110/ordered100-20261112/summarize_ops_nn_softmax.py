"""Audited finite complete ops-nn API coverage; no invented old timing control."""
import pathlib,json,hashlib,datetime
P=pathlib.Path(__file__).resolve().parent
SCOPE={538, 646, 647}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize():
 audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());dep=json.loads((P/'OPS_NN_SOFTMAX_DEPLOYMENT.json').read_text());q=P/'server2-alignment-20261113/PRIVATE_SOFTMAX_QUALIFICATION.json';qualification=json.loads(q.read_text());assert qualification['private_original_device_kernels_verified'] is True
 rows=[];seen=set()
 for cp in sorted(P.glob('rollout-ops-nn-softmax-*.CHECKPOINT.json')):
  e=json.loads(cp.read_text());label=cp.name.removesuffix('.CHECKPOINT.json');a=audit[label];assert a['checkpoint_sha256']==sha(cp)
  for r in e['batch']['results']:
   i=r['ordinal'];assert i in SCOPE and i not in seen;seen.add(i)
   if r.get('status')!='raw_checked_pending_guard':continue
   assert a['guard_valid'];checked=[x for x in a['rows'] if x['ordinal']==i];assert len(checked)==1 and checked[0]['status']=='formal_raw_and_full_output_verified'
   source=P/'remote/runs'/e['tag']/'SOURCE.json';assert sha(source)==e['batch']['source_sha256'];selected=r['sources']
   for name in ['ops_nn_softmax_api.py','ops_nn_softmax_cached_fidelity.py']:
    assert [v for k,v in selected.items() if k.endswith('/'+name)]==[dep['files'][name]]
   assert [v for k,v in selected.items() if k.endswith('/PRIVATE_SOFTMAX_QUALIFICATION.json')]==[sha(q)]
   assert e['batch']['runtime']['torch']=='2.13.0+cpu' and r['timing']['measurement_protocol']=='ascend950-isolated-ffts-v3'
   rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'source_sha256':e['batch']['source_sha256'],'selected_sources':selected,'runtime':e['batch']['runtime'],'protocol':r['timing']['measurement_protocol'],'payload_sha256':r['payload_sha256'],'prepare_sha256':r['prepare_sha256'],'reference_policy':r['reference_policy'],'full_checker_15raw_guard_valid':True,'guard_checker_15raw_valid':True,'private_device_qualification_sha256':sha(q),'original_to_new_speedup':None,'not_final_under3_ratio_closure':True})
 report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_original_APIs':len(rows),'rows':rows,'coverage_only_not_paired_performance':True,'historical_or_public_ratios_not_updated':True}
 for f in [P/'OPS_NN_SOFTMAX_CURRENT.json',P/'RESOLVED_OPS_NN_SOFTMAX.json',P.parent/'delivery/ordered_ops_nn_softmax/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix('.summary.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(f)
 return report
if __name__=='__main__':print('ACCEPTED_COMPLETE_ORIGINAL_APIS',summarize()['accepted_original_APIs'])
