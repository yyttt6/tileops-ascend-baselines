"""Close only placement-affected cases; keep original failures and raw immutable."""
import pathlib,json,hashlib,statistics,sys,types,importlib.util,datetime
P=pathlib.Path(__file__).resolve().parent;R='/opt/baseline-over3-20261110/ordered100-20261112';M=P/'remote'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('full32',P.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser);pkg=types.ModuleType('tools');pkg.kernel_task_profile=parser;sys.modules['tools']=pkg;sys.modules['tools.kernel_task_profile']=parser
from restricted_profiler.partition import partition
ledger=read(P/'LEDGER.json');checks={}
for name in ['canonical-rope-first','canonical-rope-rest','canonical-fp8-edges','canonical-catlass']:
 e=read(P/(name+'.CANONICAL.json'));assert e['guard']['performance_valid'] and e['guard']['workload_exit_code']==0
 for r in e['batch']['results']:
  assert r['status']=='verified';checks[r['ordinal']]=(e,r)
recovered=[]
for i in [41,42,44,45,46]:
 old=read(P/(ledger[str(i)]['label']+'.CHECKPOINT.json'));e,r=checks[i];B=M/'runs'/old['tag'];C=M/'runs'/e['tag'];D=B/str(i);receipt=read(C/('check-'+str(i)+'.json'));meta=read(M/'prepared'/str(i)/'prepare.json')
 assert old['guard']['performance_valid'] and old['guard']['workload_exit_code']==0 and old['guard']['phase']=='finished'
 assert meta['payload_sha256']==r['payload_sha256'] and receipt['inputs_preserved'] and receipt['independent_cpu_rotation_with_device_tables'] is True
 sources=read(B/'SOURCE.json');assert all(sources[R+'/workspace/'+k]==v for k,v in receipt['source_sha256'].items())
 left=read(B/'LOADED_LIBRARIES.json');identity_dir=C if (C/'LOADED_LIBRARIES.json').exists() else M/'runs'/read(P/'canonical-rope-rest.CANONICAL.json')['tag'];right=read(identity_dir/'LOADED_LIBRARIES.json');common=left.keys()&right.keys();assert len(common)>=10 and all(left[k]==right[k] for k in common)
 stored=read(D/'session/partition.json');raw=next(x for x in (D/'session/raw').rglob('*') if x.is_dir() and parser.trace_digest(x)==stored['trace_sha256']);pieces,proof=partition(raw,15);proof['samples']=pieces;assert proof==stored;assert parser.trace_digest(raw)==proof['trace_sha256']
 samples=[x['latency_ms'] for x in pieces];assert len(samples)==15 and all(x>0 for x in samples)
 recovered.append({'ordinal':i,'key':r['key'],'status':'original15raw_recovered_after_canonical_placement_check','device_us':statistics.mean(samples)*1000,'latency_samples_ms':samples,'protocol':'experimental-baseline-full32-marker-session-v1','original_run':old['tag'],'original_archive_sha256':old['archive_sha256'],'raw_sha256':proof['trace_sha256'],'canonical_run':e['tag'],'canonical_receipt_sha256':sha(C/('check-'+str(i)+'.json')),'payload_sha256':r['payload_sha256'],'source_all_provider_hashes_equal':True,'matching_loaded_libraries':len(common),'loaded_library_identity_check_run':identity_dir.name,'host_samples_not_reconstructed':True,'performance_windows_added':0})
# Host metric for case96 was not persisted before the old comparison failed.
# A new canonical run supplies its original15 samples rather than inventing wall time.
e=read(P/'rollout-canonical96.CHECKPOINT.json');audit=read(P/'ROLLOUT_AUDIT.json')['rollout-canonical96'];r=e['batch']['results'][0];assert r['ordinal']==96 and audit['rows'][0]['status']=='formal_raw_and_full_output_verified'
backup=P/'RESULT.before-canonical-placement-closure.json';report=read(backup if backup.exists() else P/'RESULT.json')
if not backup.exists():backup.write_bytes((P/'RESULT.json').read_bytes())
report.update(status='first100_closed_with_canonical_reference_placement',fresh_valid_baselines=59,valid_total=100,failed_pending=0,reference_placement_closure={'previous_six_failures_not_genuine_baseline_failures':True,'recovered_rope':recovered,'catlass96_new_original_isolated15':{'checkpoint':'rollout-canonical96.CHECKPOINT.json','result_sha256':audit['rows'][0]['result_sha256'],'device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic']},'additional_original_mandatory_probes':'canonical-fp8-edges.CANONICAL.json','kernel_precision_tolerance_inputs_changed':False,'historical_failed_attempts_preserved':True},failed_cases_remain_pending_no_same_condition_retry=[])
report['gate']['all100_succeeded']=True;report['result_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
# Keep first experiment denominator unchanged; report follow-up time separately.
report['canonical_followup_guarded_seconds']=sum(read(P/(n+'.CANONICAL.json'))['wall_seconds'] for n in ['canonical-rope-first','canonical-rope-rest','canonical-fp8-edges','canonical-catlass'])+e['guarded_seconds']
report['operating_throughput']['scope']='initial cohort, before separate canonical placement closure; matched exact5 result unaffected'
for item in recovered:report['selections'].append({'ordinal':item['ordinal'],'key':item['key'],'selection':'original','device_us':item['device_us'],'evidence':'FIRST100_CANONICAL_CLOSURE.json','historical_ratio_update':False})
report['selections'].append({'ordinal':96,'key':r['key'],'selection':'original','device_us':r['device_us'],'host_us_diagnostic':r['host_us_diagnostic'],'metric':r['metric'],'evidence':'rollout-canonical96.CHECKPOINT.json','historical_ratio_update':False});report['selections'].sort(key=lambda r:r['ordinal']);assert len(report['selections'])==100 and len({x['ordinal'] for x in report['selections']})==100
(P/'FIRST100_CANONICAL_CLOSURE.json').write_text(json.dumps(report['reference_placement_closure'],indent=2));(P/'RESULT.json').write_text(json.dumps(report,indent=2));(P/'SELECTION.json').write_text(json.dumps(report['selections'],indent=2));(P/'FAILED_PENDING.before-canonical-placement.json').write_bytes((P/'FAILED_PENDING.json').read_bytes()) if not (P/'FAILED_PENDING.before-canonical-placement.json').exists() else None;(P/'FAILED_PENDING.json').write_text('[]\n')
print('FIRST100',report['valid_total'],'valid;13 selected adapter improvements;gate',report['gate']['passed'])
