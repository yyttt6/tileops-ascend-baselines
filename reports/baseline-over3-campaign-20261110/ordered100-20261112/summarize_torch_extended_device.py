"""Accept fresh complete alternate API only; no invented old performance control."""
import pathlib,json,hashlib,datetime,shutil
P=pathlib.Path(__file__).resolve().parent
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def summarize():
 audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());proof=json.loads((P/'TORCH_EXTENDED_DEVICE_CPU_PROOF.json').read_text());rows=[];ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text())
 paths=list(P.glob('rollout-vllm-torch-extended-device-*.CHECKPOINT.json'))
 seen=set()
 for path in paths:
  label=path.name.removesuffix('.CHECKPOINT.json');event=json.loads(path.read_text());assert event['batch']['runtime']['torch']=='2.13.0+cpu'
  for r in event['batch']['results']:
   i=r['ordinal'];assert i in proof['scope'];assert i not in seen;seen.add(i)
   if r.get('status')!='raw_checked_pending_guard':continue
   a=audit[label];assert a['checkpoint_sha256']==sha(path) and a['guard_valid'];checked=[x for x in a['rows'] if x['ordinal']==i];assert len(checked)==1 and checked[0]['status']=='formal_raw_and_full_output_verified'
   # Frozen source contract, preparation/payload, runtime, and new native entry are source-bound.
   source=json.loads((P/'remote/runs'/event['tag']/'SOURCE.json').read_text());assert sha(P/'remote/runs'/event['tag']/'SOURCE.json')==event['batch']['source_sha256']
   selected=r['sources'];native=[v for k,v in selected.items() if k.endswith('/workloads/ascend_extended.py')];assert native==[proof['original_workload_sha256']]
   helper=[v for k,v in selected.items() if k.endswith('/torch_extended_device_reference.py')];assert helper==[proof['helper_sha256']]
   review=json.loads((P/'remote/runs'/event['tag']/str(i)/'TORCH_DEVICE_BOUNDARY.json').read_text());assert review['original_CPU_oracle_unchanged'] and review['full_reverse_AST_identical'];assert 'Missing named task coverage records' in ledger[str(i)]['result']['error']
   rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'checkpoint':path.name,'checkpoint_sha256':sha(path),'source_sha256':event['batch']['source_sha256'],'selected_sources':selected,'runtime':event['batch']['runtime'],'protocol':r['timing']['measurement_protocol'],'payload_sha256':r['payload_sha256'],'prepare_sha256':r['prepare_sha256'],'reference_policy':r['reference_policy'],'full_checker_15raw_guard_valid':True,'guard_checker_15raw_valid':True,'original_control':'Original CPUoracle neverused as devicetiming control; no speedupclaim','original_to_new_speedup':None,'alternate_contract':review,'not_final_under3_ratio_closure':True})
 report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_original_APIs':len(rows),'rows':rows,'candidate_only_not_paired_performance':True,'old_guard_and_failures_retained':True,'historical_or_public_ratios_not_updated':True}
 for f in [P/'TORCH_EXTENDED_DEVICE_CURRENT.json',P/'RESOLVED_TORCH_EXTENDED_DEVICE.json',P.parent/'delivery/ordered_torch_extended_device/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(f)
 return report
if __name__=='__main__':print('ACCEPTED_COMPLETE_ALTERNATE_APIS',summarize()['accepted_original_APIs'])
