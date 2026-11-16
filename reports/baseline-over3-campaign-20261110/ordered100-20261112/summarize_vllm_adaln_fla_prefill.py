"""Accept fresh complete alternate API only; no invented old performance control."""
import pathlib,json,hashlib,datetime,shutil
P=pathlib.Path(__file__).resolve().parent
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def summarize():
 audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());proof=json.loads((P/'VLLM_ADALN_FLA_DEFINED_CPU_PROOF.json').read_text());rows=[];ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text())
 for i in [1120,1121]:
  label='rollout-vllm-adaln-fla-prefill-'+str(i)+'-v1';path=P/(label+'.CHECKPOINT.json')
  if not path.exists():continue
  event=json.loads(path.read_text());matches=[r for r in event['batch']['results'] if r['ordinal']==i];assert len(matches)==1;r=matches[0]
  if r.get('status')!='raw_checked_pending_guard':continue
  a=audit[label];assert a['checkpoint_sha256']==sha(path) and a['guard_valid'];checked=[x for x in a['rows'] if x['ordinal']==i];assert len(checked)==1 and checked[0]['status']=='formal_raw_and_full_output_verified'
  # Frozen source contract, preparation/payload, runtime, and new native entry are source-bound.
  source=json.loads((P/'remote/runs'/event['tag']/'SOURCE.json').read_text());assert sha(P/'remote/runs'/event['tag']/'SOURCE.json')==event['batch']['source_sha256']
  selected=r['sources'];native=[v for k,v in selected.items() if k.endswith('/vllm_ascend/ops/triton/fla/layernorm_guard.py')];assert native==[proof['alternate_native_sha256']]
  contract=[v for k,v in selected.items() if k.endswith('/vllm/contracts.py')];assert contract==[proof['global_contract_sha256_unchanged']]
  helper=[v for k,v in selected.items() if k.endswith('/isolated_wrappers/vllm_adaln_fla_defined_api.py')];assert helper==[sha(P/'workspace/isolated_wrappers/vllm_adaln_fla_defined_api.py')]
  review=json.loads((P/'remote/runs'/event['tag']/('ALTERNATE_API_CONTRACT_REVIEW-'+r['key'][2]+'.json')).read_text());assert review['original_decision'] in ('supported_unverified','deployment_blocked','adapter_blocked');assert 'MLIRCompilationError' in ledger[str(i)]['result']['error'] and 'hivm-plan-memory' in ledger[str(i)]['result']['error']
  rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'checkpoint':path.name,'checkpoint_sha256':sha(path),'source_sha256':event['batch']['source_sha256'],'selected_sources':selected,'runtime':event['batch']['runtime'],'protocol':r['timing']['measurement_protocol'],'payload_sha256':r['payload_sha256'],'prepare_sha256':r['prepare_sha256'],'reference_policy':r['reference_policy'],'full_checker_15raw_guard_valid':True,'guard_checker_15raw_valid':True,'original_control':'Old distinct LayerNorm route compilePlanMemory failure retained; new completeAPI only, no oldtiming reused','original_to_new_speedup':None,'alternate_contract':review,'not_final_under3_ratio_closure':True})
 report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_alternate_APIs':len(rows),'rows':rows,'candidate_only_not_paired_performance':True,'old_guard_and_failures_retained':True,'historical_or_public_ratios_not_updated':True}
 for f in [P/'VLLM_ADALN_FLA_PREFILL_CURRENT.json',P/'RESOLVED_ADALN_FLA_PREFILL_API.json',P.parent/'delivery/ordered_vllm_adaln_fla_prefill/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(f)
 helper=P/'workspace/isolated_wrappers/vllm_adaln_fla_defined_api.py';dest=P.parent/'delivery/ordered_vllm_adaln_fla_prefill'/helper.name
 if dest.exists():assert dest.read_bytes()==helper.read_bytes()
 else:shutil.copyfile(helper,dest)
 return report
if __name__=='__main__':print('ACCEPTED_COMPLETE_ALTERNATE_APIS',summarize()['accepted_alternate_APIs'])
