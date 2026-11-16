"""Pair original/candidate in restored matched runtime; no historical ratio edits."""
import pathlib,json,hashlib,datetime,shutil
P=pathlib.Path(__file__).resolve().parent

def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def summarize():
 audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());out=[]
 for i in range(1136,1152):
  old=f'rollout-vllm-bmm-linear-candidate-{i}-original-v1';new=f'rollout-vllm-bmm-linear-candidate-{i}-linear-v1'
  if not all((P/(x+'.CHECKPOINT.json')).exists() for x in [old,new]):continue
  events=[json.loads((P/(x+'.CHECKPOINT.json')).read_text()) for x in [old,new]];rr=[e['batch']['results'][0] for e in events]
  if any(r.get('status')!='raw_checked_pending_guard' for r in rr):continue
  for name,e in zip([old,new],events):
   assert audit[name]['checkpoint_sha256']==sha(P/(name+'.CHECKPOINT.json')) and audit[name]['guard_valid']
   assert audit[name]['rows'][0]['status']=='formal_raw_and_full_output_verified'
  a,b=rr;assert events[0]['batch']['runtime']==events[1]['batch']['runtime']
  for key in ['ordinal','key','payload_sha256','prepare_sha256','metric','reference_policy']:assert a[key]==b[key],key
  assert a['timing']['measurement_protocol']==b['timing']['measurement_protocol']
  source=[json.loads((P/'remote/runs'/e['tag']/'SOURCE.json').read_text()) for e in events];common=set(source[0])&set(source[1]);assert all(source[0][k]==source[1][k] for k in common),'unexpected sharedsource change'
  gain=100*(1-b['device_us']/a['device_us']);selected='alternate' if gain>=1 else 'original'
  out.append({'ordinal':i,'key':a['key'],'original_us':a['device_us'],'candidate_us':b['device_us'],'device_reduction_percent':gain,'host_original_us_diagnostic':a['host_us_diagnostic'],'host_candidate_us_diagnostic':b['host_us_diagnostic'],'selection':selected,'selection_note':'retainoriginal for observed improvement below1percent; no claim of statistical significance','original_checkpoint':old+'.CHECKPOINT.json','candidate_checkpoint':new+'.CHECKPOINT.json','checkpoints_sha256':[sha(P/(x+'.CHECKPOINT.json')) for x in [old,new]],'source_sha256':[e['batch']['source_sha256'] for e in events],'common_SOURCE_entries_identical':len(common),'expected_unique_source_paths':{'original':sorted(set(source[0])-set(source[1])),'candidate':sorted(set(source[1])-set(source[0]))},'guard_checker_15raw_valid':True,'runtime':events[0]['batch']['runtime'],'protocol':a['timing']['measurement_protocol'],'selected_sources':rr[1 if selected=='alternate' else 0]['sources']})
 report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete_pairs':len(out),'rows':out,'effective_retest_coverage_not_incremented':True,'historical_or_public_ratios_not_updated':True,'runtime_repair':'REFERENCE_RUNTIME_PRIVATE_COPY.json','not_all_SOURCE_bytes_same_expected_import_API_differences_explicit':True}
 for path in [P/'VLLM_BMM_LINEAR_CURRENT.json',P.parent/'delivery/ordered_vllm_bmm_linear_api/ROUTES.json']:
  path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
 stage=P.parent/'delivery/ordered_vllm_bmm_linear_api';helper=P/'workspace/isolated_wrappers/vllm_bmm_linear_api.py';dest=stage/helper.name
 if dest.exists():assert dest.read_bytes()==helper.read_bytes()
 else:shutil.copyfile(helper,dest)
 return report
if __name__=='__main__':print('COMPLETE_VLLM_BMM_LINEAR_PAIRS',summarize()['complete_pairs'])
