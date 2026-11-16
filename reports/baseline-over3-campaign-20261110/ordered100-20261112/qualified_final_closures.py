"""CPU-only closure counter; never counts estimates or unqualified ratios."""
from pathlib import Path
import json,hashlib,datetime,math
def collect(P,now):
 P=Path(P);f=P/'FINAL_SAME_ENV_CLOSURES_CURRENT.json'
 if not f.exists():return []
 data=json.loads(f.read_text());assert data['scope_total']==2119;found={};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for row in data['rows']:
  under=row.get('final_under3_qualified') is True;retained=row.get('final_retained_over3_qualified') is True
  if not (under or retained):continue
  assert row['original_full_checker_15raw_source_guard_and_denominator_identity_valid'] is True and row['protocol']=='ascend950-isolated-ffts-v3'
  if row['ordinal'] in {1356,1358}:
   assert row.get('frozen_checker_supplement_verified') is True
   supplement=P/row['frozen_checker_supplement_record'];assert sha(supplement)==row['frozen_checker_supplement_sha256'];cert=json.loads(supplement.read_text());assert cert['status']=='passed' and cert['archive_both_local_and_remote_SHA_verified']
   guard=P/cert['guard_receipt'];assert sha(guard)==cert['guard_receipt_sha256'];g=json.loads(guard.read_text());assert g['phase']=='finished' and g['performance_valid'] and g['workload_exit_code']==0
   for role in ['baseline','tileops']:assert cert['qualified_checkpoint_sha256'][row[role+'_checkpoint']]==row[role+'_checkpoint_sha256']
  reuse=row.get('tileops_result_ordinal',row['ordinal'])!=row['ordinal']
  if reuse:
   assert row['ordinal'] in {390,391,392,393} and row['no_native_denominator_repeat']
   cf=P/row['tileops_reuse_record'];assert sha(cf)==row['tileops_reuse_record_sha256'];cert=json.loads(cf.read_text());assert cert['status']=='verified_same_case_inputs_frozen_checker_runtime_protocol_source_and_binary' and cert['baseline_ordinal']==row['ordinal'] and cert['tileops_result_ordinal']==row['tileops_result_ordinal'] and cert['actual_denominator_library']=='TileOps' and cert['no_native_denominator_repeat'] and cert['no_checkpoint_key_or_raw_rewrite']
   for role in ['baseline','tileops']:assert cert[role+'_checkpoint']==row[role+'_checkpoint'] and cert[role+'_checkpoint_sha256']==row[role+'_checkpoint_sha256']
  events=[];results=[]
  for role in ['baseline','tileops']:
   cp=P/row[role+'_checkpoint'];assert sha(cp)==row[role+'_checkpoint_sha256'];e=json.loads(cp.read_text());g=e['guard'];assert g['phase']=='finished' and g['performance_valid'] is True and g['workload_exit_code']==0;result_ordinal=row.get('tileops_result_ordinal',row['ordinal']) if role=='tileops' else row['ordinal'];r=next(r for r in e['batch']['results'] if r['ordinal']==result_ordinal);assert r['status']=='raw_checked_pending_guard' and r['timing']['measurement_protocol']==row['protocol'];events.append(e);results.append(r)
  b,t=events;br,tr=results
  if reuse:
   assert br['key'][1:]==tr['key'][1:] and tr['measured_library']=='TileOps'
   metadata=[]
   for result,ordinal,label in [(br,row['ordinal'],'baseline'),(tr,row['tileops_result_ordinal'],'tileops')]:
    file=P/'remote/prepared-rollout'/str(ordinal)/'prepare.json';assert sha(file)==result['prepare_sha256']==cert[label+'_prepare_sha256'];metadata.append(json.loads(file.read_text()))
   for key in ['params','verification','input_seed','reference_policy','torch','payload_sha256','sources']:assert metadata[0][key]==metadata[1][key]
  else:
   for key in ['key','prepare_sha256']:assert br[key]==tr[key]
  for key in ['payload_sha256','reference_policy','metric']:assert br[key]==tr[key]
  assert b['batch']['runtime']==t['batch']['runtime'] and b['guard']['boot_id']==t['guard']['boot_id'] and b['guard']['device_node']==t['guard']['device_node']
  ratio=br['device_us']/tr['device_us'];assert math.isfinite(ratio) and abs(ratio-row['ratio'])<1e-12 and ((under and ratio<3) or (retained and ratio>=3))
  qf=P/row['tileops_generated_identity_record'];assert sha(qf)==row['tileops_generated_identity_record_sha256'];q=json.loads(qf.read_text());assert q['full_original_public_Op_kernel_generated_binary_checker_raw_guard_identity_verified'] and q['archive_both_local_and_remote_SHA_verified']
  at=datetime.datetime.fromisoformat(row['completed_at']);assert at.tzinfo is not None
  if now-datetime.timedelta(seconds=3600)<=at<=now:
   found[row['ordinal']]={'ordinal':row['ordinal'],'completed_at':row['completed_at'],'under3':under,'retained_over3':retained,'ratio':ratio,'summary':f.name,'baseline_checkpoint':row['baseline_checkpoint'],'tileops_checkpoint':row['tileops_checkpoint']}
 return list(found.values())
