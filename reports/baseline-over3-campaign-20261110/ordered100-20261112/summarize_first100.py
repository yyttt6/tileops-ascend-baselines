import pathlib,json,statistics,hashlib,datetime
import sys,subprocess
if (pathlib.Path(__file__).resolve().parent/'FIRST100_CANONICAL_CLOSURE.json').exists():
 subprocess.run([sys.executable,str(pathlib.Path(__file__).resolve().parent/'close_first100.py')],check=True)
 raise SystemExit(0)

P=pathlib.Path(__file__).resolve().parent;ROOT=P.parent
ledger=json.loads((P/'LEDGER.json').read_text());candidates=json.loads((P/'CANDIDATE_LEDGER.json').read_text());audits=json.loads((P/'AUDIT.json').read_text());assert len(ledger)==100
verified={(label,r['ordinal']) for label,b in audits.items() for r in b['rows'] if r['status']=='formal_raw_and_full_output_verified'}
cp={p.name.removesuffix('.CHECKPOINT.json'):json.loads(p.read_text()) for p in P.glob('*.CHECKPOINT.json')}
selected=[];fail=[];reused=[];fresh=[]
for key,entry in sorted(ledger.items(),key=lambda kv:int(kv[0])):
 r=entry['result'];i=int(key)
 if r['status'].startswith('historical_valid'):
  reused.append(i);selected.append({'ordinal':i,'key':r['key'],'selection':'original_reused','device_us':r['device_us'],'evidence':'ATVC_REUSE_REVIEW.json'});continue
 if r['status']!='raw_checked_pending_guard':fail.append({'ordinal':i,'key':r['key'],'status':r['status'],'phase':r['phase'],'error':r['error'],'evidence':entry['label']+'.CHECKPOINT.json'});continue
 assert (entry['label'],i) in verified;fresh.append(i);choice=r;which='original';selection_evidence=entry['label']+'.CHECKPOINT.json';delta=None
 if key in candidates:
  alt=candidates[key];a=alt['result'];assert (alt['label'],i) in verified
  assert a['payload_sha256']==r['payload_sha256'] and a['prepare_sha256']==r['prepare_sha256']
  assert cp[alt['label']]['batch']['runtime']==cp[entry['label']]['batch']['runtime']
  delta=100*(1-a['device_us']/r['device_us'])
  if a['device_us']<r['device_us']:choice=a;which='zero_pad_adapter';selection_evidence=alt['label']+'.CHECKPOINT.json'
 selected.append({'ordinal':i,'key':r['key'],'selection':which,'device_us':choice['device_us'],'original_device_us':r['device_us'],'device_reduction_percent':delta,'host_us_diagnostic':choice['host_us_diagnostic'],'metric':r['metric'],'evidence':selection_evidence,'protocol':choice['timing'].get('measurement_protocol','original-isolated'),'historical_ratio_update':False})
old=[cp['old-tilekernels-'+str(i)+'-'+str(i)] for i in range(5,10)];new=cp['new-tilekernels-5-9'];before=sum(e['guarded_seconds'] for e in old);after=new['guarded_seconds'];drift=[]
for i,e in zip(range(5,10),old):
 a=e['batch']['results'][0];b=next(r for r in new['batch']['results'] if r['ordinal']==i);assert a['payload_sha256']==b['payload_sha256'] and a['prepare_sha256']==b['prepare_sha256'] and a['topology']==b['topology'];assert e['batch']['runtime']==new['batch']['runtime']
 drift.append({'ordinal':i,'old_us':a['device_us'],'new_us':b['device_us'],'device_change_percent':100*(b['device_us']/a['device_us']-1),'source_input_topology_equal':True})
history=json.loads((ROOT/'throughput/EXISTING_TIMING_ANALYSIS.json').read_text());old_formal=[r for r in history['rows'] if r['mode']=='formal'];old_seconds=sum(r['guard_seconds'] for r in old_formal);old_rate=len(old_formal)*3600/old_seconds
# Count all fresh production attempts, including failed rows and the SDK mismatch,
# not reused41 and not the deliberate matched experiment controls.
events=[json.loads(s) for s in (P/'TIMING.jsonl').read_text().splitlines()];production=[e for e in events if not e['label'].startswith('old')];new_seconds=sum(e['guarded_seconds'] for e in production);valid_legs=len(fresh)+sum(a['result']['status']=='raw_checked_pending_guard' for a in candidates.values());new_rate=valid_legs*3600/new_seconds
speed=before/after;gate=speed>=1.5 and len(selected)+len(fail)==100
report={'status':'first100_reviewed','total':100,'fresh_valid_baselines':len(fresh),'reused_valid_baselines':len(reused),'valid_total':len(selected),'failed_pending':len(fail),'new_candidates_selected':sum(r['selection']=='zero_pad_adapter' for r in selected),'candidate_reduction_range_percent':[min(r['device_reduction_percent'] for r in selected if r['selection']=='zero_pad_adapter'),max(r['device_reduction_percent'] for r in selected if r['selection']=='zero_pad_adapter')],'matched_same5_fp8_cases':{'old_processes':5,'new_processes':1,'old_guarded_seconds':before,'new_guarded_seconds':after,'speedup':speed,'reduction_percent':100*(1-after/before),'15windows_each_case_both_paths':True,'case_device_comparison':drift,'cold_compile_not_removed':'old control5 contains first compile, controls6..9 and shared treatment use persistent cache; do not attribute whole reduction solely to profiler'},'operating_throughput':{'historical_formal_valid_legs':len(old_formal),'historical_guard_seconds':old_seconds,'historical_valid_legs_per_hour':old_rate,'current_fresh_valid_legs_including_candidates':valid_legs,'current_all_production_guard_seconds_including_failed_attempts':new_seconds,'current_valid_legs_per_hour':new_rate,'ratio':new_rate/old_rate,'not_a_matched_causal_comparison':'different workload mix; source reuse and comparison controls excluded; CPU prep, engineering and external idle reported separately'},'gate':{'required_speedup':1.5,'passed':gate,'basis':'matched exact5cases + all valid rows full output/15raw/validguard','all100_succeeded':not fail},'continuation_authorized':True,'continue_remaining':gate,'failed_cases_remain_pending_no_same_condition_retry':fail,'selections':selected,'public_history_or_ratios_modified':False,'reference_checker_tolerance_unchanged':True,'result_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(P/'RESULT.json').write_text(json.dumps(report,indent=2));(P/'SELECTION.json').write_text(json.dumps(selected,indent=2));(P/'FAILED_PENDING.json').write_text(json.dumps(fail,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('selections','failed_cases_remain_pending_no_same_condition_retry')},indent=2))
