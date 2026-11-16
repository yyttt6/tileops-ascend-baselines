"""CPU-only exact-route staging; never modifies shared provider or published ratios."""
import pathlib,json,hashlib,shutil,datetime
P=pathlib.Path(__file__).resolve().parent
D=P.parent/'delivery/ordered_sgl_wrappers';D.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
alts=json.loads((P/'SGL_WRAPPER_CANDIDATES.json').read_text());ledger=json.loads((P/'ROLLOUT_LEDGER.json').read_text());proof=[]
for i,a in alts.items():
 if a['status']!='raw_checked_pending_guard':continue
 base=ledger[i];left_path=P/(base['label']+'.CHECKPOINT.json');right_path=P/(a['label']+'.CHECKPOINT.json');left=json.loads(left_path.read_text());right=json.loads(right_path.read_text());b=base['result'];v=a['result']
 assert left['guard']['performance_valid'] and right['guard']['performance_valid'] and left['exit_code']==right['exit_code']==0
 assert left['batch']['runtime']==right['batch']['runtime']
 assert b['payload_sha256']==v['payload_sha256'] and b['prepare_sha256']==v['prepare_sha256']
 assert all(v['sources'].get(k)==s for k,s in b['sources'].items())
 assert b['metric']==v['metric']=='device_async'
 original=next(x for x in left['batch']['results'] if x['ordinal']==int(i))
 for r in [original,v]:
  assert r['check']['checked_outputs']==r['check']['total_outputs']>0 and not r['check']['unchecked_reason']
  assert len(r['timing']['latency_samples_ms'])==15
 assert original['timing']['measurement_protocol']==v['timing']['measurement_protocol']
 extras={k:v['sources'][k] for k in v['sources'].keys()-b['sources'].keys()};assert len(extras)==1
 for k,h in extras.items():
  name=pathlib.Path(k).name;source=P/'workspace/isolated_wrappers'/name;assert sha(source)==h
  target=D/name
  if target.exists():assert sha(target)==h
  else:shutil.copyfile(source,target)
 gain=1-v['device_us']/b['device_us'];assert a['candidate_selected']==(gain>=.03)
 proof.append({'ordinal':int(i),'key':v['key'],'selection':a['selection'],'original_us':b['device_us'],'candidate_us':v['device_us'],'device_reduction_percent':100*gain,'protocol':v['timing']['measurement_protocol'],'original_checkpoint':left_path.name,'original_checkpoint_sha256':sha(left_path),'candidate_checkpoint':right_path.name,'candidate_checkpoint_sha256':sha(right_path),'input_prepare_runtime_provider_source_equal':True,'candidate_helper_sources':extras,'both_guards_valid':True,'full_checker_and_15raw_each_leg':True})
for name,value in [('ROUTES.json',proof),('SUMMARY.json',{'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'valid_candidate_pairs':len(proof),'candidate_selected':sum(x['selection']=='reviewed_wrapper_candidate' for x in proof),'staged_only':True,'historical_or_public_ratio_update':False})]:
 f=D/name
 if f.exists():f.with_name(name+'.before-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).write_bytes(f.read_bytes())
 tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(f)
(D/'DELIVERY.txt').write_text('Exact routes in ROUTES.json only. Original complete native APIs and wrapping retained. Lerp midpoint replaces the complete original mean-producing API; Neg/Silu use only reviewed zero-padding skips when separately tested. Full checker/15raw, identical input/runtime/provider sources and both valid guards required. Original retained below predeclared3percent observed device gain. Experimental session protocol explicitly retained; no mixed historical denominator or website update.\n')
print('STAGED',len(proof),'pairs;',sum(x['selection']=='reviewed_wrapper_candidate' for x in proof),'candidates selected')
