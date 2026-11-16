from pathlib import Path
import json,hashlib,datetime,importlib.util
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';M=D/'remote';R=Path('/opt/baseline-over3-20261110/ordered100-20261112');ROOT=D.parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('full32',D.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
def summarize():
 audit=json.loads((D/'ROLLOUT_AUDIT.json').read_text());rows=[];proofs=[]
 for cp in sorted(D.glob('rollout-fla-cube-api-sdk-*-v4.CHECKPOINT.json')):
  e=json.loads(cp.read_text());a=audit.get(e['label']);assert a and a['guard_valid'] and a['checkpoint_sha256']==sha(cp);assert e['guard']['phase']=='finished' and e['guard']['workload_exit_code']==0 and e['guard']['performance_valid'];base=M/'runs'/e['tag'];assert sha(base/'SOURCE.json')==e['batch']['source_sha256']
  for r in e['batch']['results']:
   if r['status']!='raw_checked_pending_guard':continue
   assert r['ordinal'] in set(json.loads((D/'FLA_CUBE_API_SCOPE.json').read_text())['ordinals']) and r['key'][0]=='flash-linear-attention-npu' and r['key'][1] in ('GemmFwdOp','BmmFwdOp','GroupedGemmFwdOp');assert any(x['ordinal']==r['ordinal'] and x['status']=='formal_raw_and_full_output_verified' for x in a['rows']);cap=base/str(r['ordinal'])/'JIT_CAPTURE';receipt=json.loads((cap/'RETENTION.json').read_text());assert receipt['original_AoT_source_build_package_identity_verified']
   for name,v in receipt['files'].items():assert sha(cap/name)==v['sha256']
   kernels={}
   for rec in receipt['records']:
    meta=rec['metadata'];assert sha(cap/rec['object_path'])==sha(cap/rec['build_object_path'])==meta['sha256'];assert sha(cap/rec['original_CPP_path'])==sha(cap/rec['compiled_CPP_path'])==rec['original_CPP_sha256']==sha(ROOT/'ascend_baselines/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/chunk_fwd_h.cpp')
    names=[meta['kernelName'],*[x['kernelName'] for x in meta['kernelList']]]
    for name in names:kernels[name]=meta['sha256'];kernels[name+'__kernel0']=meta['sha256']
   assert receipt['loaded_library'] in (cap/'loaded-maps.txt').read_text();t=r['timing'];assert t['measurement_protocol']=='ascend950-isolated-ffts-v3' and len(t['retained_profiles'])==len(t['latency_samples_ms'])==15;windows=[]
   for prof,value in zip(t['retained_profiles'],t['latency_samples_ms'],strict=True):
    raw=M/Path(prof['profile_path']).relative_to(R);assert parser.trace_digest(raw)==prof['profile_trace_sha256'];p=parser.parse_profile(raw,1);assert p['profile_named_task_coverage_complete'] and abs(p['latency_ms']-value)<1e-9;bound=[x for x in p['profile_kernel_tasks'] if x['name'] in kernels];assert bound,('No original AoT kernel identity',sorted(set(x['name'] for x in p['profile_kernel_tasks'])));windows.append({'trace_sha256':prof['profile_trace_sha256'],'all_tasks':p['profile_kernel_tasks'],'device_us':p['latency_ms']*1000})
   rows.append({'ordinal':r['ordinal'],'key':r['key'],'api':r['api'],'device_us':r['device_us'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'protocol':t['measurement_protocol'],'payload_sha256':r['payload_sha256'],'full_checker_15raw_guard_valid':True,'original_AoT_source_build_package_profile_all15_identity':True,'not_paired_or_final_ratio':True});proofs.append({'ordinal':r['ordinal'],'checkpoint_sha256':sha(cp),'guard_valid':True,'dual_archive_sha256':e['archive_sha256'],'kernels':kernels,'all15_windows_complete':windows,'retention_sha256':sha(cap/'RETENTION.json')})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_complete_APIs':len(rows),'rows':rows,'staged_not_published':True}
 for p in [D/'FLA_CUBE_API_CURRENT.json',D/'RESOLVED_FLA_CUBE_API.json']:
  tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+chr(10));tmp.replace(p)
 (A/'FLA_CUBE_API_AOT_ATTRIBUTION_CURRENT.json').write_text(json.dumps({'checked_at':out['checked_at'],'proofs':proofs},indent=2)+chr(10));return out
if __name__=='__main__':print(summarize()['accepted_complete_APIs'])
