from pathlib import Path
import json,hashlib,datetime,importlib.util,fcntl,os
D=Path(__file__).resolve().parent;M=D/'remote';R=Path('/opt/baseline-over3-20261110/ordered100-20261112');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();spec=importlib.util.spec_from_file_location('full32adaln7',D.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
def summarize():
 rows=[];proofs=[]
 for cp in sorted(D.glob('rollout-tla-adaln7-math-alternative-*-h1-v*.CHECKPOINT.json')):
  e=json.loads(cp.read_text())
  if any(r.get('status')!='raw_checked_pending_guard' for r in e.get('batch',{}).get('results',[])):continue
  assert e['exit_code']==0 and e['guard']['phase']=='finished' and e['guard']['performance_valid'] and e['guard']['workload_exit_code']==0;assert sha(D/(e['tag']+'.tar.gz'))==e['archive_sha256'];base=M/'runs'/e['tag'];assert sha(base/'SOURCE.json')==e['batch']['source_sha256'];r=e['batch']['results'][0];i=r['ordinal'];assert i in range(1617,1624) and r['status']=='raw_checked_pending_guard' and r['reference_policy']=='original_npu';assert r['check']['checked_outputs']==r['check']['total_outputs']==1 and not r['check']['unchecked_reason'];cap=base/str(i)/'JIT_CAPTURE';ret=json.loads((cap/'RETENTION.json').read_text());q=json.loads((cap/'ORIGINAL_NORM_HOST_QUALIFICATION.json').read_text());assert q['full_API_embedded_kernel_source_build_guard_and_dual_archive_verified']
  for name,h in ret['files'].items():assert sha(cap/name)==h
  maps=(cap/'loaded-maps.txt').read_text();assert ret['loaded_host_path'] in maps and sha(cap/'libops_math_layernorm.so')==ret['loaded_host_sha256']==q['build_record']['binary_sha256'];assert sha(cap/'layer_norm.asc')==q['original_source_sha256'];assert any(ret['qualified_original_kernel_name'] in v['text'] for v in q['kernel_launch_evidence']);packages=json.loads((cap/'MUL_ADD_RETENTION.json').read_text());kernel_maps={}
  for label,data in packages.items():
   folder=cap/label;manifest=json.loads((folder/data['manifest']).read_text());assert sha(folder/data['manifest'])==data['manifest_sha256']
   for name,h in data['files'].items():assert sha(folder/'package'/name)==h
   bindings={}
   for name,h in data['device_objects'].items():
    f=folder/'package'/name;assert sha(f)==h and f.read_bytes()[:4]==b'\x7fELF';md=json.loads(f.with_suffix('.json').read_text());assert md['sha256']==h
    for item in md['kernelList']:bindings[item['kernelName']]=h
   kernel_maps[label]=bindings
   assert data['package_root'] in maps,'Exact private library package must be loaded'
  timing=r['timing'];assert timing['measurement_protocol']=='ascend950-isolated-ffts-v3' and len(timing['retained_profiles'])==15;windows=[];zero=r['key'][1]=='AdaLayerNormZeroFwdOp'
  for prof,ms in zip(timing['retained_profiles'],timing['latency_samples_ms'],strict=True):
   raw=M/Path(prof['profile_path']).relative_to(R);assert parser.trace_digest(raw)==prof['profile_trace_sha256'];v=parser.parse_profile(raw,1);assert v['profile_named_task_coverage_complete'] and abs(v['latency_ms']-ms)<1e-9;names=[x['name'] for x in v['profile_kernel_tasks']];assert names.count(ret['qualified_original_kernel_name'])==1;mul=[{'name':n,'ELF_sha256':kernel_maps['Mul'][n]} for n in names if n in kernel_maps['Mul']];add=[{'name':n,'ELF_sha256':kernel_maps['Add'][n]} for n in names if n in kernel_maps['Add']];assert len(mul)==(2 if zero else 1) and len(add)==1,(i,names);windows.append({'trace_sha256':prof['profile_trace_sha256'],'complete_device_ms':ms,'LayerNorm_source_bound':True,'Mul':mul,'Add':add,'all_named_device_tasks':names})
  proofs.append({'ordinal':i,'checkpoint_sha256':sha(cp),'all15_full32_windows':windows,'LayerNorm_qualification_sha256':sha(cap/'ORIGINAL_NORM_HOST_QUALIFICATION.json'),'Mul_Add_retention_sha256':sha(cap/'MUL_ADD_RETENTION.json'),'archive_sha256':e['archive_sha256']});rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'payload_sha256':r['payload_sha256'],'reference_policy':r['reference_policy'],'full_checker_15raw_guard_valid':True,'original_TLA_implementation_not_qualified':True,'explicit_complete_Math_API_alternative':True,'not_paired_or_finalratio':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_complete_APIs':len(rows),'rows':rows,'new_pairs':0,'new_final_closures':0,'staged_not_published':True};updates={'TLA_ADALN7_MATH_ALTERNATIVE_CURRENT.json':out,'RESOLVED_TLA_ADALN7_MATH_ALTERNATIVE.json':out,'server2-alignment-20261113/TLA_ADALN7_MATH_ALTERNATIVE_ATTRIBUTION.json':{'checked_at':out['checked_at'],'proofs':proofs}}
 with (D.parents[2]/'.agent_memory/.maintenance.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);back=D/'memory-history'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');back.mkdir(parents=True)
  for name,value in updates.items():
   p=D/name;old=p.read_bytes() if p.exists() else None
   if old is not None:(back/p.name).write_bytes(old);(back/(p.name+'.sha256')).write_text(hashlib.sha256(old).hexdigest())
   assert (p.read_bytes() if p.exists() else None)==old;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));os.replace(t,p)
 return out
if __name__=='__main__':print('ACCEPTED',summarize()['accepted_complete_APIs'])
