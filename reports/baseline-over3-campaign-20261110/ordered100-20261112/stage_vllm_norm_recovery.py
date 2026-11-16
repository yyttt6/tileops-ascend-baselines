"""Stage exact accepted alternative APIs; original failed evidence is never overwritten."""
import pathlib,json,hashlib,datetime,shutil
P=pathlib.Path(__file__).resolve().parent;D=P.parent/'delivery/ordered_vllm_norm_recovery';D.mkdir(exist_ok=True);ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text());audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());routes=[];overlay={}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for cp in sorted(P.glob('rollout-vllm-norm-candidate-*.CHECKPOINT.json')):
 e=json.loads(cp.read_text());assert audit[e['label']]['checkpoint_sha256']==sha(cp)
 assert e['guard']['performance_valid'] and e['exit_code']==0
 for r in e['batch']['results']:
  if r['status']!='raw_checked_pending_guard':continue
  i=str(r['ordinal']);old=ledger[i];assert old['result']['status']=='failed';assert 'MLIRCompilationError' in old['result']['error']
  originalcp=P/(old['label']+'.CHECKPOINT.json');oe=json.loads(originalcp.read_text());assert oe['batch']['runtime']==e['batch']['runtime']
  meta=json.loads((P/'remote/prepared-rollout'/i/'prepare.json').read_text());assert meta['payload_sha256']==r['payload_sha256'];assert sha(P/'remote/prepared-rollout'/i/'prepare.json')==r['prepare_sha256']
  assert r['check']['checked_outputs']==r['check']['total_outputs']>0 and not r['check']['unchecked_reason'];assert len(r['timing']['latency_samples_ms'])==15
  copied={}
  for source,h in r['sources'].items():
   if '/isolated_wrappers/' not in source:continue
   f=P/'workspace/isolated_wrappers'/pathlib.Path(source).name;assert sha(f)==h
   target=D/f.name
   if target.exists():assert sha(target)==h
   else:shutil.copyfile(f,target)
   copied[f.name]=h
  assert copied
  row={'ordinal':int(i),'key':r['key'],'selection':'complete_same_pinned_vendor_alternative_API','candidate_us':r['device_us'],'original_compile_failure':originalcp.name,'original_failure_checkpoint_sha256':sha(originalcp),'accepted_checkpoint':cp.name,'accepted_checkpoint_sha256':sha(cp),'api':r['api'],'protocol':r['timing']['measurement_protocol'],'full_checker_15raw_guard_valid':True,'helpers':copied,'current_valid_selection_only_no_old_new_speedup':True}
  routes.append(row);overlay[i]=row
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
for f,value in [(D/'ROUTES.json',routes),(P/'RESOLVED_ALTERNATIVE_APIS.json',{'updated_at':now,'rows':overlay,'coverage_rule':'Only union with original valid ordinals; never doublecount same case; source failures preserved in original ledger'})]:
 if f.exists():f.with_name(f.name+'.before-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).write_bytes(f.read_bytes())
 tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(f)
(D/'DELIVERY.txt').write_text('Exact accepted GroupNorm/InstanceNorm routes only. Original complete vendor LayerNorm API failed compilation; a second complete API from the same locked vendor is selected after full checker,15raw replay and valid guard. No old/new speedup claimed without valid old timing. Original grouping, reduction domain, weight/bias expansion, dtype, epsilon, all copies and output shape retained. Staged only; no shared registry or website edits.\n')
print('STAGED',len(routes),'valid alternative selections; original failures retained')
