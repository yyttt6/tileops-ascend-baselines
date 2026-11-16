from pathlib import Path
import json,hashlib,datetime,importlib.util
D=Path(__file__).resolve().parent;A=D/'server2-alignment-20261113';M=D/'remote';R=Path('/opt/baseline-over3-20261110/ordered100-20261112');ROOT=D.parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('full32',D.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
def summarize():
 import fla_tap_audit_reuse as reuse
 catalog=reuse.load()
 for digest,path in catalog['anchors'].items():assert reuse.sha(D/path)==digest,'Prior complete audit anchor changed'
 audits=json.loads((D/'ROLLOUT_AUDIT.json').read_text());rows=[];proofs=[]
 for cp in sorted(D.glob('rollout-fla-tap-api-sdk-*-v[78].CHECKPOINT.json')):
  e=json.loads(cp.read_text())
  if not all(x['status']=='raw_checked_pending_guard' for x in e['batch']['results']):continue
  a=audits.get(e['label']);assert a and a['guard_valid'] and a['checkpoint_sha256']==sha(cp);assert e['guard']['phase']=='finished' and e['guard']['workload_exit_code']==0 and e['guard']['performance_valid'];base=M/'runs'/e['tag'];assert sha(base/'SOURCE.json')==e['batch']['source_sha256']
  for r in e['batch']['results']:
   if r['status']!='raw_checked_pending_guard':continue
   cert=catalog['certificates'].get(str(r['ordinal']))
   if cert is not None:
    row,proof=reuse.verify(cp,e,a,cert,parser,ROOT);rows.append(row);proofs.append(proof);continue
   assert r['ordinal'] in json.loads((D/'FLA_TAP_API_SCOPE.json').read_text())['ordinals'] and r['key'][0]=='flash-linear-attention-npu' and r['key'][1] in ('INT8DequantPerTensorFwdOp','BmmFwdOp','GroupedGemmFwdOp','SSDStatePassingFwdOp','SSDChunkCouplingFwdOp','SSDChunkCumsumFwdOp');assert any(x['ordinal']==r['ordinal'] and x['status']=='formal_raw_and_full_output_verified' for x in a['rows']);cap=base/str(r['ordinal'])/'JIT_CAPTURE';receipt=json.loads((cap/'RETENTION.json').read_text());assert receipt['actual_source_cache_key_loaded_binary_launches_verified']
   for name,v in receipt['files'].items():assert sha(cap/name)==v['sha256']
   records=receipt['records']
   for key,rec in records.items():
    assert key==rec['hash']==rec['recomputed_hash'];assert sha(cap/rec['source_copy'])==rec['source_sha256']==sha(ROOT/'ascend_baselines/flash-linear-attention-npu/fla/ops/triton/triton_core'/Path(rec['source_file']).name);assert sha(cap/rec['compiler_copy'])==rec['compiler_sha256'];assert any(sha(cap/rel)==rec['binary_sha256'] for rel in rec['retained_files'].values())
   aot_kernels={}
   if r['key'][1] in ('SSDStatePassingFwdOp','SSDChunkCouplingFwdOp','SSDChunkCumsumFwdOp'):
    acap=cap/'AOT';ar=json.loads((acap/'RETENTION.json').read_text());assert ar['original_AoT_source_build_package_identity_verified'] and receipt['aot_retention_sha256']==sha(acap/'RETENTION.json')
    for name,v in ar['files'].items():assert sha(acap/name)==v['sha256']
    for rec in ar['records']:
     meta=rec['metadata'];assert sha(acap/rec['object_path'])==sha(acap/rec['build_object_path'])==meta['sha256'];assert sha(acap/rec['original_CPP_path'])==sha(acap/rec['compiled_CPP_path'])==rec['original_CPP_sha256']==sha(ROOT/'ascend_baselines/flash-linear-attention-npu/fla/ops/ascendc'/('kda/kda_gate_cumsum/op_kernel/kda_gate_cumsum.cpp' if r['key'][1]=='SSDChunkCumsumFwdOp' else 'gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/chunk_fwd_h.cpp'))
     for name in [meta['kernelName'],*[x['kernelName'] for x in meta['kernelList']]]:aot_kernels[name]=meta['sha256'];aot_kernels[name+'__kernel0']=meta['sha256']
    assert aot_kernels and ar['loaded_library'] in (acap/'loaded-maps.txt').read_text()
   t=r['timing'];assert t['measurement_protocol']=='ascend950-isolated-ffts-v3' and len(t['retained_profiles'])==len(t['latency_samples_ms'])==15;windows=[]
   for index,(prof,value) in enumerate(zip(t['retained_profiles'],t['latency_samples_ms'],strict=True),4):
    raw=M/Path(prof['profile_path']).relative_to(R);assert parser.trace_digest(raw)==prof['profile_trace_sha256'];p=parser.parse_profile(raw,1);assert p['profile_named_task_coverage_complete'] and abs(p['latency_ms']-value)<1e-9;launches=receipt['formal_calls'][str(index)];assert launches;names=set(x['name'] for x in p['profile_kernel_tasks'])
    if aot_kernels:assert names & set(aot_kernels),'No original AoT ChunkFwdH binding in actual window'
    for launch in launches:
     rec=records[launch['hash']];assert launch['name']==rec['function_name'];allowed={rec['kernel_name'],rec['function_name'],rec['kernel_name']+'__kernel0',rec['function_name']+'__kernel0'};assert names&allowed,('Actual loaded Triton kernel not bound to profile',names,allowed)
    windows.append({'trace_sha256':prof['profile_trace_sha256'],'actual_loaded_ELF_launches':launches,'all_tasks':p['profile_kernel_tasks'],'device_us':p['latency_ms']*1000})
   rows.append({'ordinal':r['ordinal'],'key':r['key'],'api':r['api'],'device_us':r['device_us'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'payload_sha256':r['payload_sha256'],'protocol':t['measurement_protocol'],'full_checker_15raw_guard_valid':True,'original_Triton_source_recomputed_key_actual_loaded_ELF_launches_all15_verified':True,'not_paired_or_final_ratio':True});proofs.append({'ordinal':r['ordinal'],'checkpoint_sha256':sha(cp),'guard_valid':True,'dual_archive_sha256':e['archive_sha256'],'retention_sha256':sha(cap/'RETENTION.json'),'all15_windows_complete':windows,'compiled_kernels':records,'aot_kernels':aot_kernels,'aot_retention_sha256':receipt.get('aot_retention_sha256')})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_complete_APIs':len(rows),'rows':rows,'staged_not_published':True}
 for p in [D/'FLA_TAP_API_CURRENT.json',D/'RESOLVED_FLA_TAP_API.json']:
  tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+chr(10));tmp.replace(p)
 proofpath=A/'FLA_TAP_API_TRITON_ATTRIBUTION_CURRENT.json';proofpath.write_text(json.dumps({'checked_at':out['checked_at'],'proofs':proofs,'prior_complete_audit_anchors':catalog['anchors']},indent=2)+chr(10));reuse.promote(out,proofpath);return out
if __name__=='__main__':print(summarize()['accepted_complete_APIs'])
