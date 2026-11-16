"""Matched-protocol comparisons; current qualified controls reused, no original rerun."""
from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def summarize():
 ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text());audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());rows=[]
 for i in [2002,2003]:
  base=P/(ledger[str(i)]['label']+'.CHECKPOINT.json');new=P/('rollout-tla-mul-zero-pad-'+str(i)+'-v1.CHECKPOINT.json')
  if not new.exists():continue
  b=json.loads(base.read_text());n=json.loads(new.read_text());br=next(r for r in b['batch']['results'] if r['ordinal']==i);nr=next(r for r in n['batch']['results'] if r['ordinal']==i)
  if nr['status']!='raw_checked_pending_guard':continue
  assert br['status']=='raw_checked_pending_guard' and br['key']==nr['key'] and br['payload_sha256']==nr['payload_sha256'] and br['prepare_sha256']==nr['prepare_sha256'] and br['api']==nr['api'] and br['reference_policy']==nr['reference_policy'] and br['metric']==nr['metric']
  assert b['batch']['runtime']==n['batch']['runtime'] and br['timing']['measurement_protocol']==nr['timing']['measurement_protocol']=='experimental-baseline-full32-marker-session-v1'
  assert b['guard']['boot_id']==n['guard']['boot_id'] and b['guard']['device_node']==n['guard']['device_node']
  for cp,e in [(base,b),(new,n)]:
   a=audit[cp.name.removesuffix('.CHECKPOINT.json')];assert a['checkpoint_sha256']==sha(cp) and a['guard_valid'];assert next(r for r in a['rows'] if r['ordinal']==i)['status']=='formal_raw_and_full_output_verified'
  def source(e):
   p=P/'remote/runs'/e['tag']/'SOURCE.json';assert sha(p)==e['batch']['source_sha256'];return json.loads(p.read_text())
  bs=source(b);ns=source(n);changes={k:[bs[k],ns[k]] for k in bs.keys()&ns.keys() if bs[k]!=ns[k]};allowed='/opt/baseline-over3-20261110/ordered100-20261112/workspace/TileOPs/benchmarks/ascend_all_pairs/sgl/row_gemv.py'
  assert not changes or changes=={allowed:['618ca3568073f86f64d2873967140a6f4cf3419771c549eee50e021e4e428204','9b86e94c65732d1e10f4be445113418104ad42d47436272385bcf425274d9d4e']},changes
  bl=json.loads((P/'remote/runs'/b['tag']/'LOADED_LIBRARIES.json').read_text());nl=json.loads((P/'remote/runs'/n['tag']/'LOADED_LIBRARIES.json').read_text());assert all(bl[k]==nl[k] for k in bl.keys()&nl.keys());jit=lambda libs:{k:v for k,v in libs.items() if '/.cache/all-pairs/tla/' in k};assert jit(nl) and all(jit(bl).get(k)==v for k,v in jit(nl).items()),'Candidate JIT library is absent or byte-different from original four-case batch'
  reduction=100*(1-nr['device_us']/br['device_us']);rows.append({'ordinal':i,'key':nr['key'],'guard_checker_15raw_valid':True,'original_checkpoint':base.name,'candidate_checkpoint':new.name,'original_device_us':br['device_us'],'candidate_device_us':nr['device_us'],'device_reduction_percent':reduction,'original_host_us_diagnostic':br['host_us_diagnostic'],'candidate_host_us_diagnostic':nr['host_us_diagnostic'],'selection':'candidate' if reduction>=5 else 'original_conservative_small_gain_or_regression','protocol':nr['timing']['measurement_protocol'],'comparison':'Noncontemporaneous samehost/currentruntime qualifiedcontrol reuse, no randomizedcausal or publicratio claim','candidate_JIT_library_SHAs_in_original_batch':True,'original_batch_other_JIT_libraries':sorted(set(jit(bl))-set(jit(nl))),'SOURCE_common_differences':changes,'not_final_under3_ratio_closure':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'paired_cases':len(rows),'rows':rows,'coverage_increment':0,'no_historical_or_public_ratio_rewrite':True};p=P/'TLA_MUL_ZERO_PAD_CURRENT.json';p.write_text(json.dumps(out,indent=2)+'\n');return out
