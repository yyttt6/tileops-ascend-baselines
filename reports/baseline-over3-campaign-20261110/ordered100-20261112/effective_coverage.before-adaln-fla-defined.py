"""Deduplicate accepted originals and explicitly audited selected-API repairs."""
import pathlib,json,datetime
P=pathlib.Path(__file__).resolve().parent
def snapshot():
 ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text());original={int(k) for k,v in ledger.items() if v.get('result',{}).get('status')=='raw_checked_pending_guard'};valid=set(original);layers={}
 for name in ['RESOLVED_ALTERNATIVE_APIS.json','RESOLVED_SOURCE_PATHS.json','RESOLVED_TLA_UNPACK.json']:
  f=P/name
  if not f.exists():continue
  rows=json.loads(f.read_text())['rows'];rows=rows.values() if isinstance(rows,dict) else rows;accepted=set()
  for r in rows:
   okay=r.get('full_checker_15raw_guard_valid') or (r.get('guard_valid') and r.get('result',{}).get('status')=='raw_checked_pending_guard')
   if okay:accepted.add(int(r['ordinal']))
  layers[name]={'accepted':len(accepted),'new_outside_original_ledger':len(accepted-original),'ordinals':sorted(accepted)};valid|=accepted
 assert all(100<=i<2119 for i in valid)
 return {'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_total':2119,'first100_closed':100,'ledger_original_valid':len(original),'selected_or_original_valid_union_after100':len(valid),'effective_coverage':100+len(valid),'percentage':100*(100+len(valid))/2119,'unresolved_recorded':len(set(map(int,ledger))-valid),'remaining_unvisited':2019-len(ledger),'layers':layers,'accepted_ordinals_after100':sorted(valid),'full_alternative_API_review_not_assumed_complete':True,'public_or_historical_ratios_not_updated':True}
if __name__=='__main__':
 s=snapshot();f=P/'CURRENT_VALID_COVERAGE.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(f);print(json.dumps({k:v for k,v in s.items() if k not in ('accepted_ordinals_after100','layers')}))
