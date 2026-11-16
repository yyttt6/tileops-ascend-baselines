from pathlib import Path
import json,hashlib,datetime,importlib.util
D=Path(__file__).resolve().parent;M=D/'remote';R=Path('/opt/baseline-over3-20261110/ordered100-20261112');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();spec=importlib.util.spec_from_file_location('full32var',D.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
def summarize():
 rows=[];proofs=[]
 for cp in sorted(D.glob('rollout-tla-mean-alternative-*-h1-v2.CHECKPOINT.json')):
  e=json.loads(cp.read_text())
  assert e['exit_code']==0 and e['guard']['performance_valid'] and e['guard']['workload_exit_code']==0;base=M/'runs'/e['tag'];assert sha(D/(e['tag']+'.tar.gz'))==e['archive_sha256'];source=json.loads((base/'SOURCE.json').read_text());assert sha(base/'SOURCE.json')==e['batch']['source_sha256']
  for r in e['batch']['results']:
   i=r['ordinal'];assert i in [1969] and r['status']=='raw_checked_pending_guard';assert r['check']['checked_outputs']==r['check']['total_outputs']==(2 if r['key'][1]=='VarMeanFwdOp' else 1) and not r['check']['unchecked_reason'];cap=base/str(i)/'JIT_CAPTURE';ret=json.loads((cap/'RETENTION.json').read_text());pm=json.loads((cap/'PACKAGE_MANIFEST.json').read_text());sm=json.loads((cap/'SOURCE_MANIFEST.json').read_text());assert sm['pin']=='64322f6487162c0934ec0364af57ed2f51175a03';assert sha(cap/'PACKAGE_MANIFEST.json')==ret['package_manifest_sha256'] and sha(cap/'SOURCE_MANIFEST.json')==ret['source_manifest_sha256']
   for name,v in pm['files'].items():assert sha(cap/'PACKAGE'/name)==v['sha256']
   original=D.parents[2]/'ascend_baselines/ops-math'
   for name,v in sm['files'].items():assert sha(original/name)==v['sha256']
   kernels={}
   for item in ret['records']:
    md=item['metadata'];mfile=cap/'PACKAGE'/item['metadata_relative'];assert json.loads(mfile.read_text())==md;obj=mfile.with_name(md['binFileName']+md['binFileSuffix']);assert sha(obj)==md['sha256']==item['device_ELF_sha256']
    for k in md['kernelList']:kernels[k['kernelName']]=md['sha256']
   assert sha(cap/'PACKAGE/op_api/lib/libcust_opapi.so')==ret['loaded_host_sha256'];assert ret['loaded_host_path'] in (cap/'loaded-maps.txt').read_text();timing=r['timing'];assert len(timing['retained_profiles'])==len(timing['latency_samples_ms'])==15;windows=[]
   for p,value in zip(timing['retained_profiles'],timing['latency_samples_ms']):
    raw=M/Path(p['profile_path']).relative_to(R);assert parser.trace_digest(raw)==p['profile_trace_sha256'];t=parser.parse_profile(raw,1);assert t['profile_named_task_coverage_complete'] and abs(t['latency_ms']-value)<1e-9;bound=[x for x in t['profile_kernel_tasks'] if x['name'] in kernels];assert bound,'Actual Mean kernel not bound to original private AoT metadata';windows.append({'trace_sha':p['profile_trace_sha256'],'actual_kernels':[{**x,'device_ELF_sha':kernels[x['name']]} for x in bound],'full32_complete':True})
   proof={'ordinal':i,'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'all15_actual_kernel_to_original_AoT':windows,'source_pin':sm['pin'],'source_files_verified':len(sm['files']),'retention_sha256':sha(cap/'RETENTION.json'),'guard_valid':True,'archive_sha256':e['archive_sha256'],'original_outputs_checked':r['check']};proofs.append(proof);rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'source_attribution':'ops-math alternative completeAPI for originalTLAcontract','checkpoint':cp.name,'checkpoint_sha256':sha(cp),'payload_sha256':r['payload_sha256'],'full_checker_15raw_guard_valid':True,'actual_original_AoT_source_device_profile_identity_verified':True,'not_paired_or_finalratio':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_complete_APIs':len(rows),'rows':rows,'staged_not_published':True}
 for name in ['TLA_MEAN_ALTERNATIVE_API_CURRENT.json','RESOLVED_TLA_MEAN_ALTERNATIVE_API.json']:
  p=D/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(p)
 (D/'server2-alignment-20261113/TLA_MEAN_ALTERNATIVE_AOT_ATTRIBUTION.json').write_text(json.dumps({'checked_at':out['checked_at'],'proofs':proofs},indent=2));return out
