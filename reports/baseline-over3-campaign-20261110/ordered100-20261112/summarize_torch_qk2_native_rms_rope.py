from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def summarize():
 originals={r['ordinal']:r for r in json.loads((P/'TORCH_QK2_ORIGINAL_CURRENT.json').read_text())['rows']};audit=json.loads((P/'ROLLOUT_AUDIT.json').read_text());rows=[]
 for cp in sorted(P.glob('rollout-torch-qk2-native-rms-rope-*.CHECKPOINT.json')):
  e=json.loads(cp.read_text());nr=e['batch']['results'][0];i=nr['ordinal'];assert i in {1356,1358}
  if nr['status']!='raw_checked_pending_guard':continue
  original=P/originals[i]['checkpoint'];b=json.loads(original.read_text());br=next(r for r in b['batch']['results'] if r['ordinal']==i)
  assert br['status']=='raw_checked_pending_guard'
  for key in ['key','payload_sha256','prepare_sha256','reference_policy','metric']:assert br[key]==nr[key],key
  assert b['batch']['runtime']==e['batch']['runtime'] and b['guard']['boot_id']==e['guard']['boot_id'] and b['guard']['device_node']==e['guard']['device_node'];assert br['timing']['measurement_protocol']==nr['timing']['measurement_protocol']=='ascend950-isolated-ffts-v3'
  maps=[];libs=[]
  for f,x in [(original,b),(cp,e)]:
   a=audit[f.name.removesuffix('.CHECKPOINT.json')];assert a['checkpoint_sha256']==sha(f) and a['guard_valid'] and next(z for z in a['rows'] if z['ordinal']==i)['status']=='formal_raw_and_full_output_verified';sf=P/'remote/runs'/x['tag']/'SOURCE.json';assert sha(sf)==x['batch']['source_sha256'];maps.append(json.loads(sf.read_text()));libs.append(json.loads((P/'remote/runs'/x['tag']/'LOADED_LIBRARIES.json').read_text()))
  assert any(k.endswith('/worker_torch_qk2_native_rms_rope.py') and v==sha(P/'worker_torch_qk2_native_rms_rope.py') for k,v in maps[1].items())
  differences={k:[maps[0][k],maps[1][k]] for k in maps[0].keys()&maps[1].keys() if maps[0][k]!=maps[1][k]};assert not differences,differences
  assert all(libs[0][k]==libs[1][k] for k in libs[0].keys()&libs[1].keys());assert nr.get('candidate_complete_native_RMS_ROPE_QK_program') and not nr['measured_original_Torch_program']
  reduction=100*(1-nr['device_us']/br['device_us']);rows.append({'ordinal':i,'key':nr['key'],'guard_checker_15raw_valid':True,'original_checkpoint':original.name,'candidate_checkpoint':cp.name,'original_device_us':br['device_us'],'candidate_device_us':nr['device_us'],'device_reduction_percent':reduction,'selection':'candidate' if reduction>=5 else 'original_conservative_small_gain_or_regression','original_host_us_diagnostic':br['host_us_diagnostic'],'candidate_host_us_diagnostic':nr['host_us_diagnostic'],'protocol':nr['timing']['measurement_protocol'],'common_SOURCE_differences':differences,'comparison':'Samehost/currentruntime original then candidate; matched complete inputs/protocol/boot/source; original complete FP32 QK program versus same complete program with native FP32 RMS and interleaved rotary_mul; both RMS outputs, timed coefficient expansion, clone V and supplied table retained, no randomizedcausal claim','not_final_under3_ratio_closure':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'paired_cases':len(rows),'rows':rows,'coverage_increment':0,'no_historical_or_public_ratio_rewrite':True}
 for f in [P/'TORCH_QK2_NATIVE_RMS_ROPE_CURRENT.json',P.parent/'delivery/ordered_torch_qk2_native_rms_rope/ROUTES.json']:
  f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(out,indent=2)+'\n')
 return out
