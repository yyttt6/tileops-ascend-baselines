from pathlib import Path
import json,hashlib,datetime,importlib.util
D=Path(__file__).resolve().parent;R=Path('/opt/baseline-over3-20261110/ordered100-20261112');M=D/'remote';A=D/'server2-alignment-20261113';ROOT=D.parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('full32',D.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
def summarize():
 audits=json.loads((D/'ROLLOUT_AUDIT.json').read_text());rows=[];proofs=[]
 for cp in sorted(D.glob('rollout-ops-transformer-state-api-sdk-*-v1.CHECKPOINT.json')):
  e=json.loads(cp.read_text());a=audits.get(e['label'])
  if not a:continue
  assert a['checkpoint_sha256']==sha(cp) and a['guard_valid'];assert e['guard']['phase']=='finished' and e['guard']['workload_exit_code']==0 and e['guard']['performance_valid'];base=M/'runs'/e['tag'];source=json.loads((base/'SOURCE.json').read_text());assert sha(base/'SOURCE.json')==e['batch']['source_sha256']
  for r in e['batch']['results']:
   i=r['ordinal'];assert i in {805} and r['key'][:2]==['ops-transformer','SSDStatePassingFwdOp']
   if r['status']!='raw_checked_pending_guard':continue
   assert any(x['ordinal']==i and x['status']=='formal_raw_and_full_output_verified' for x in a['rows']);cap=base/str(i)/'JIT_CAPTURE';receipt=json.loads((cap/'RETENTION.json').read_text());assert receipt['original_CPP_source_and_object_loadedso_deviceELFs_verified']
   for name,v in receipt['files'].items():assert sha(cap/name)==v['sha256']
   identity=json.loads((cap/'FORMAL_BUILD_IDENTITY.json').read_text());build=json.loads((cap/'BUILD_RECIPE_RESULT.json').read_text());assert build['compile_exit_code']==build['link_exit_code']==0;assert sha(cap/'BUILD_RECIPE_RESULT.json')==identity['build_result_sha256'];assert sha(cap/'original_transformer_state.o')==identity['object_sha256'];assert sha(cap/'liboriginal_state_passing.so')==identity['binary_sha256']==receipt['loaded_library_sha256'];assert str(Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-state-original-v1')/'liboriginal_state_passing.so') in (cap/'loaded-maps.txt').read_text()
   for name,expected in identity['original_source_sha256'].items():assert sha(cap/name)==expected;assert sha(ROOT/'ascend_baselines/ops-transformer'/name.removeprefix('source/'))==expected
   symbol=identity['kernel_symbol'];assert symbol=='kernel_cust_chunk_state_passing';assert all(symbol.encode() in (cap/Path(name).relative_to(Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-state-original-v1'))).read_bytes() for name in identity['device_ELFs']);timing=r['timing'];assert timing['measurement_protocol']=='ascend950-isolated-ffts-v3' and len(timing['retained_profiles'])==len(timing['latency_samples_ms'])==15;windows=[]
   for prof,value in zip(timing['retained_profiles'],timing['latency_samples_ms'],strict=True):
    path=M/Path(prof['profile_path']).relative_to(R);assert parser.trace_digest(path)==prof['profile_trace_sha256'];p=parser.parse_profile(path,1);assert p['profile_named_task_coverage_complete'] and abs(p['latency_ms']-value)<1e-9;bound=[t for t in p['profile_kernel_tasks'] if symbol in t['name']];assert len(bound)==r['original_kernel_calls_per_invocation'],('Everywindow requires exact original completefusion devicekernel',sorted(set(t['name'] for t in p['profile_kernel_tasks'])));windows.append({'trace_sha256':prof['profile_trace_sha256'],'all_tasks':p['profile_kernel_tasks'],'device_us':p['latency_ms']*1000})
   proofs.append({'ordinal':i,'checkpoint_sha256':sha(cp),'guard_valid':True,'loaded_library_sha256':identity['binary_sha256'],'actual_device_ELFs':identity['device_ELFs'],'actual_private_entry':symbol,'all15_windows_complete':windows,'dual_archive_sha256':e['archive_sha256'],'source_and_originalflags_unchanged':True});rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'payload_sha256':r['payload_sha256'],'protocol':timing['measurement_protocol'],'full_checker_15raw_guard_valid':True,'original_CPP_object_loadedso_deviceELF_all15_binding':True,'not_paired_or_final_ratio':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_complete_APIs':len(rows),'rows':rows,'staged_not_published':True}
 for file in [D/'OPS_TRANSFORMER_STATE_API_CURRENT.json',D/'RESOLVED_OPS_TRANSFORMER_STATE_API.json']:
  tmp=file.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(file)
 (A/'TRANSFORMER_STATE_API_STATIC_ATTRIBUTION_CURRENT.json').write_text(json.dumps({'checked_at':out['checked_at'],'proofs':proofs},indent=2)+'\n');return out
if __name__=='__main__':print(summarize()['accepted_complete_APIs'])
