"""Select complete qualified current-contract APIs conservatively; never repeat measurements."""
from pathlib import Path
import json,hashlib,datetime,copy
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def summarize():
 audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());groups=[]
 for name in ['TORCH_QK2_NATIVE_RMS_CURRENT.json','TORCH_QK2_NATIVE_RMS_ROPE_CURRENT.json']:
  f=P/name
  if f.exists():groups.append((name,json.loads(f.read_text())))
 selected={}
 for name,group in groups:
  for row in group['rows']:
   assert row['ordinal'] in {1356,1358} and row['guard_checker_15raw_valid']
   for field in ['original_checkpoint','candidate_checkpoint']:
    f=P/row[field];e=json.loads(f.read_text());a=audit[f.name.removesuffix('.CHECKPOINT.json')];assert a['checkpoint_sha256']==sha(f) and a['guard_valid'] and e['guard']['performance_valid'];assert next(x for x in a['rows'] if x['ordinal']==row['ordinal'])['status']=='formal_raw_and_full_output_verified'
   r=copy.deepcopy(row);r['selected_pair_summary']=name;actual=r['candidate_device_us'] if r['selection']=='candidate' else r['original_device_us'];prior=selected.get(r['ordinal'])
   if prior is None or actual<prior['selected_device_us']*.95:r['selected_device_us']=actual;selected[r['ordinal']]=r
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':[selected[i] for i in sorted(selected)],'paired_cases':len(selected),'conservative_selection_requires_additional5percent_device_reduction':True,'complete_current_contract_only':True,'source_summaries':{name:sha(P/name) for name,_ in groups},'not_final_closure_until_denominator_identity_and_ratio_valid':True}
 f=P/'TORCH_QK2_BEST_CURRENT.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(f);print('QK_BEST',[(r['ordinal'],r['selected_pair_summary'],r['selected_device_us']) for r in out['rows']],flush=True);return out
if __name__=='__main__':summarize()
