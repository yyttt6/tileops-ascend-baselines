"""CPU replay exact retained ELF/private preprocessor origin + all15 full32 traces."""
from pathlib import Path
import json,hashlib,datetime,importlib.util,posixpath,re
D=Path(__file__).resolve().parent;R=Path('/opt/baseline-over3-20261110/ordered100-20261112');M=D/'remote';ROOT=D.parents[2];A=D/'server2-alignment-20261113';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('full32',D.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
def summarize():
 audits=json.loads((D/'ROLLOUT_AUDIT.json').read_text());rows=[];proofs=[]
 prior=json.loads((D/'OPS_TRANSFORMER_COUPLING_API_CURRENT.json').read_text())['rows'];capsules=[]
 saved={x['ordinal']:x for x in json.loads((A/'TRANSFORMER_COUPLING_API_JIT_ATTRIBUTION_CURRENT.json').read_text())['proofs']};accepted_cp={}
 for old in prior:
  cp0=D/old['checkpoint'];e0=json.loads(cp0.read_text());pr=saved[old['ordinal']];assert sha(cp0)==old['checkpoint_sha256']==pr['checkpoint_sha256'];assert pr['guard_valid'] and audits[e0['label']]['guard_valid'] and audits[e0['label']]['checkpoint_sha256']==sha(cp0);assert pr['dual_archive_sha256']==e0['archive_sha256'];assert len(pr['all15_windows_complete'])==15;assert e0['guard']['performance_valid'] and e0['guard']['phase']=='finished' and e0['guard']['workload_exit_code']==0;rows.append(old);proofs.append(pr);accepted_cp[cp0.name]=sha(cp0)
 for old in prior:
  cp0=D/old['checkpoint'];assert sha(cp0)==old['checkpoint_sha256'];e0=json.loads(cp0.read_text());assert audits[e0['label']]['guard_valid'] and e0['guard']['performance_valid'];cap0=M/'runs'/e0['tag']/str(old['ordinal'])/'JIT_CAPTURE'
  if all('reused_qualified_capsule' in x for x in json.loads((cap0/'RETENTION.json').read_text())['records']):continue
  capsules.append((e0,cap0,cp0.name,sha(cp0)))
 for cp in sorted(D.glob('rollout-ops-transformer-coupling-api-sdk-*-v1.CHECKPOINT.json')):
  if cp.name in accepted_cp:
   assert sha(cp)==accepted_cp[cp.name];continue
  e=json.loads(cp.read_text());a=audits.get(e['label'])
  if not a:continue
  assert a['checkpoint_sha256']==sha(cp) and a['guard_valid'];assert e['guard']['performance_valid'] and e['guard']['workload_exit_code']==0 and e['guard']['phase']=='finished';base=M/'runs'/e['tag'];source=json.loads((base/'SOURCE.json').read_text());assert sha(base/'SOURCE.json')==e['batch']['source_sha256']
  for r in e['batch']['results']:
   i=r['ordinal'];assert i in {804} and r['key'][:2]==['ops-transformer','SSDChunkCouplingFwdOp']
   if r['status']!='raw_checked_pending_guard':continue
   assert any(x['ordinal']==i and x['status']=='formal_raw_and_full_output_verified' for x in a['rows']);cap=base/str(i)/'JIT_CAPTURE';receipt=json.loads((cap/'RETENTION.json').read_text());kernels={};origins=[]
   sets=[(cap,receipt,None)]
   for e0,cap0,cpname,cpdigest in capsules:
    if e0['guard']['device_node']!=e['guard']['device_node'] or e0['batch']['runtime']!=e['batch']['runtime']:continue
    oldr=next(x for x in e0['batch']['results'] if x['ordinal']==int(cap0.parent.name));prefix='/private-v1/vendors/arithmeticpairs_transformer/'
    assert {k:v for k,v in oldr['sources'].items() if prefix in k}=={k:v for k,v in r['sources'].items() if prefix in k}
    sets.append((cap0,json.loads((cap0/'RETENTION.json').read_text()),{'checkpoint':cpname,'sha256':cpdigest,'capsule_sha256':sha(cap0/'RETENTION.json')}))
   for cap,receipt,capsule_origin in sets:
    for item in receipt['records']:
     for name,v in item['retained_files'].items():assert sha(cap/name)==v['sha256']
     meta=item['metadata'];obj=cap/(meta['binFileName']+meta['binFileSuffix']);assert sha(obj)==meta['sha256']==item['device_ELF_sha256'];key=item['kernel']+'__kernel0';assert key not in kernels or kernels[key]==meta['sha256'];kernels[key]=meta['sha256'];found=[]
     for f in [*cap.glob(item['kernel']+'*_vec.i'),*cap.glob(item['kernel']+'*_cube.i')]:
      for line in f.read_text(errors='replace').splitlines():
       q=re.match(r'# \d+ "([^"]+)"',line)
       if q and '/private-v1/vendors/arithmeticpairs_transformer/' in q[1] and q[1].endswith('/op_kernel/grouped_matmul_add.cpp'):
        target=posixpath.normpath(q[1]);assert target in r['sources'];original=ROOT/'ascend_baselines/ops-transformer/gmm/grouped_matmul_add/op_kernel/grouped_matmul_add.cpp';assert sha(original)==r['sources'][target];found.append({'private_source':target,'source_sha256':r['sources'][target]})
     assert found,'Retained preprocessed source not bound to actual original private source';origin=found[0];origin['supplemental_qualified_capsule']=capsule_origin;origins.append(origin)
   timing=r['timing'];assert timing['measurement_protocol']=='ascend950-isolated-ffts-v3';assert len(timing['retained_profiles'])==len(timing['latency_samples_ms'])==15;windows=[]
   for prof,value in zip(timing['retained_profiles'],timing['latency_samples_ms'],strict=True):
    raw=M/Path(prof['profile_path']).relative_to(R);assert parser.trace_digest(raw)==prof['profile_trace_sha256'];p=parser.parse_profile(raw,1);assert p['profile_named_task_coverage_complete'] and abs(p['latency_ms']-value)<1e-9;bound=[t for t in p['profile_kernel_tasks'] if t['name'] in kernels];assert bound,'No exact retained private ELF kernel in profile';windows.append({'trace_sha256':prof['profile_trace_sha256'],'tasks':p['profile_kernel_tasks'],'total_device_us':p['latency_ms']*1000})
   proofs.append({'ordinal':i,'checkpoint_sha256':sha(cp),'guard_valid':True,'original_private_preprocessed_source':origins,'actual_profile_kernel_to_ELF_sha256':kernels,'all15_windows_complete':windows,'retention_receipt_sha256':sha(base/str(i)/'JIT_CAPTURE/RETENTION.json'),'dual_archive_sha256':e['archive_sha256']})
   rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'protocol':timing['measurement_protocol'],'payload_sha256':r['payload_sha256'],'prepare_sha256':r['prepare_sha256'],'full_checker_15raw_guard_valid':True,'actual_private_ELF_and_original_source_all15trace_identity_verified':True,'not_paired_or_final_ratio':True})
 catalog={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'capsules':[{'guard_device_node':json.loads((D/x['checkpoint']).read_text())['guard']['device_node'],'path':str(R/'runs'/json.loads((D/x['checkpoint']).read_text())['tag']/str(x['ordinal'])/'JIT_CAPTURE'),'checkpoint_sha256':x['checkpoint_sha256']} for x in rows if any('reused_qualified_capsule' not in item for item in json.loads((M/'runs'/json.loads((D/x['checkpoint']).read_text())['tag']/str(x['ordinal'])/'JIT_CAPTURE/RETENTION.json').read_text())['records'])]}
 (D/'TRANSFORMER_COUPLING_API_QUALIFIED_JIT_CAPSULES.json').write_text(json.dumps(catalog,indent=2)+chr(10))
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_complete_APIs':len(rows),'rows':rows,'staged_not_published':True}
 for p in [D/'OPS_TRANSFORMER_COUPLING_API_CURRENT.json',D/'RESOLVED_OPS_TRANSFORMER_COUPLING_API.json']:
  t=p.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2)+chr(10));t.replace(p)
 p=A/'TRANSFORMER_COUPLING_API_JIT_ATTRIBUTION_CURRENT.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'checked_at':out['checked_at'],'proofs':proofs},indent=2)+chr(10));t.replace(p);return out
if __name__=='__main__':
 r=summarize();print(json.dumps({'accepted':r['accepted_complete_APIs'],'results':[(x['ordinal'],x['device_us']) for x in r['rows']]}))
