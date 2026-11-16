from pathlib import Path
import json,hashlib,datetime,importlib.util,fcntl,os
D=Path(__file__).resolve().parent;M=D/'remote';R=Path('/opt/baseline-over3-20261110/ordered100-20261112');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();sp=importlib.util.spec_from_file_location('full32torch8',D.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(sp);sp.loader.exec_module(parser)
def summarize():
 rows=[];proofs=[]
 for cp in sorted(D.glob('rollout-remaining-torch-complete-*-h1-v1.CHECKPOINT.json')):
  e=json.loads(cp.read_text())
  if any(r.get('status')!='raw_checked_pending_guard' for r in e.get('batch',{}).get('results',[])):continue
  assert e['exit_code']==0 and e['guard']['phase']=='finished' and e['guard']['performance_valid'] and e['guard']['workload_exit_code']==0;assert sha(D/(e['tag']+'.tar.gz'))==e['archive_sha256'];base=M/'runs'/e['tag'];assert sha(base/'SOURCE.json')==e['batch']['source_sha256'];source=json.loads((base/'SOURCE.json').read_text());r=e['batch']['results'][0];i=r['ordinal'];assert i in {370,371,733,734,1736,1737,2101,2107};assert r['check']['checked_outputs']==r['check']['total_outputs']>0 and not r['check']['unchecked_reason'];cap=base/str(i)/'SDK_CAPTURE';sdk=json.loads((cap/'SDK_KERNEL_RETENTION.json').read_text());assert not sdk['unbound_names'],sdk['unbound_names'];native={}
  for rec in sdk['records']:
   name=rec['name']
   if rec['kind']=='SDK_AICORE':
    for match in rec['matches']:
     for key in ['metadata','device_ELF']:assert sha(cap/match[key]['retained_file'])==match[key]['sha256']
     md=json.loads((cap/match['metadata']['retained_file']).read_text());assert md['sha256']==match['device_ELF']['sha256'];assert name in {x['kernelName'] for x in md.get('kernelList',[])} or name==md.get('kernelName');assert (cap/match['device_ELF']['retained_file']).read_bytes()[:4]==b'\x7fELF'
   else:
    assert rec['kind']=='SDK_declared_AICPU_actual_raw_task'
    for config in rec['configs']:assert sha(cap/config['metadata']['retained_file'])==config['metadata']['sha256'] and name in json.loads((cap/config['metadata']['retained_file']).read_text())
   native[name]=rec
  assert sdk['loaded_runtime_libraries'] and all(f in (cap/'PROCESS_MAPS.txt').read_text() for f in sdk['loaded_runtime_libraries']);timing=r['timing'];assert timing['measurement_protocol']=='ascend950-isolated-ffts-v3' and len(timing['retained_profiles'])==15;windows=[]
  for prof,ms in zip(timing['retained_profiles'],timing['latency_samples_ms'],strict=True):
   raw=M/Path(prof['profile_path']).relative_to(R);assert parser.trace_digest(raw)==prof['profile_trace_sha256'];v=parser.parse_profile(raw,1);assert v['profile_named_task_coverage_complete'] and abs(v['latency_ms']-ms)<1e-9;names=[t['name'] for t in v['profile_kernel_tasks']];assert names and set(names)<=set(native);windows.append({'trace_sha256':prof['profile_trace_sha256'],'complete_device_ms':ms,'all_named_tasks':names})
  assert any(k.endswith('/remaining_torch_complete_alternative_api.py') and v==sha(D/'remaining_torch_complete_alternative_api.py') for k,v in source.items());proofs.append({'ordinal':i,'checkpoint_sha256':sha(cp),'actual_SDK_OPP':sdk['actual_opp'],'SDK_retention_sha256':sha(cap/'SDK_KERNEL_RETENTION.json'),'all15_full32_windows':windows,'SDK_AICORE_ELF_and_AICPU_declaration_separate':True,'archive_sha256':e['archive_sha256']});rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'payload_sha256':r['payload_sha256'],'reference_policy':r['reference_policy'],'full_checker_15raw_guard_valid':True,'explicit_complete_original_Torch_API_alternative':True,'original_library_native_provider_not_qualified':True,'not_paired_or_finalratio':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_complete_APIs':len(rows),'rows':rows,'new_pairs':0,'new_final_closures':0};updates={'REMAINING_TORCH_COMPLETE_ALTERNATIVE_CURRENT.json':out,'RESOLVED_REMAINING_TORCH_COMPLETE_ALTERNATIVE.json':out,'server2-alignment-20261113/REMAINING_TORCH_COMPLETE_ATTRIBUTION.json':{'checked_at':out['checked_at'],'proofs':proofs}}
 with (D.parents[2]/'.agent_memory/.maintenance.lock').open('a') as l:
  fcntl.flock(l,fcntl.LOCK_EX);b=D/'memory-history'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');b.mkdir(parents=True)
  for name,value in updates.items():
   p=D/name;old=p.read_bytes() if p.exists() else None
   if old is not None:(b/p.name).write_bytes(old);(b/(p.name+'.sha256')).write_text(hashlib.sha256(old).hexdigest())
   assert (p.read_bytes() if p.exists() else None)==old;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));os.replace(t,p)
 return out
