"""CPU raw replay; require actual named Cumprod AICPU spans in every isolated window."""
from pathlib import Path
import json,hashlib,datetime,importlib.util,statistics
D=Path(__file__).resolve().parent;R=Path('/opt/baseline-over3-20261110/ordered100-20261112');M=D/'remote';A=D/'server2-alignment-20261113'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('full32',D.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
def summarize():
 audit=json.loads((D/'ROLLOUT_AUDIT.json').read_text());rows=[];proofs=[]
 for cp in sorted(D.glob('rollout-ops-math-cumprod-sdk-*-v1.CHECKPOINT.json')):
  e=json.loads(cp.read_text());label=e['label'];a=audit.get(label)
  if not a:continue
  assert a['checkpoint_sha256']==sha(cp) and a['guard_valid'];assert e['guard']['phase']=='finished' and e['guard']['performance_valid'] and e['guard']['workload_exit_code']==0
  source=M/'runs'/e['tag']/'SOURCE.json';assert sha(source)==e['batch']['source_sha256'];source_map=json.loads(source.read_text());assert source_map[str(R/'worker_ops_math_cumprod.py')]==sha(D/'worker_ops_math_cumprod.py')
  libs=json.loads((M/'runs'/e['tag']/'LOADED_LIBRARIES.json').read_text());api='/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1/private-v2/vendors/allpairs_math_scanbool_math/op_api/lib/libcust_opapi.so';assert libs[api]=='c4e65bd0de342e1b5dfabe893732cb374e3b629e7a54eda2c44561697d2c1334'
  for r in e['batch']['results']:
   i=r['ordinal'];assert i in {252,253,254,255} and r['key'][0:2]==['ops-math','CumprodFwdOp']
   if r['status']!='raw_checked_pending_guard':continue
   assert any(x['ordinal']==i and x['status']=='formal_raw_and_full_output_verified' for x in a['rows']);timing=r['timing'];assert timing['measurement_protocol']=='ascend950-isolated-ffts-v3' and len(timing['retained_profiles'])==15;windows=[]
   for prof,v in zip(timing['retained_profiles'],timing['latency_samples_ms'],strict=True):
    p=M/Path(prof['profile_path']).relative_to(R);assert parser.trace_digest(p)==prof['profile_trace_sha256'];q=parser.parse_profile(p,1);tasks=q['profile_kernel_tasks'];cu=[x for x in tasks if 'Cumprod' in x['name'] and x['core_roles']==['AICPU']];assert len(cu)==1 and q['profile_named_task_coverage_complete'];assert abs(q['latency_ms']-v)<1e-9
    windows.append({'trace_sha256':prof['profile_trace_sha256'],'tasks':tasks,'total_device_us':q['latency_ms']*1000})
   proof={'ordinal':i,'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'api_loaded_sha256':libs[api],'windows':windows,'complete_all_named_work_counted_once':True,'declared_SDK_AICPU_not_private_AIcore':True};proofs.append(proof)
   rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'protocol':timing['measurement_protocol'],'source_sha256':e['batch']['source_sha256'],'prepare_sha256':r['prepare_sha256'],'payload_sha256':r['payload_sha256'],'full_checker_15raw_guard_valid':True,'actual_Cumprod_AICPU_all15_windows_verified':True,'not_final_ratio_or_paired_speedup':True})
 out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_complete_APIs':len(rows),'rows':rows,'not_published_or_final_ratio':True}
 for p in [D/'OPS_MATH_CUMPROD_CURRENT.json',D/'RESOLVED_OPS_MATH_CUMPROD.json']:
  t=p.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2)+'\n');t.replace(p)
 p=A/'CUMPROD_FULL32_AICPU_ATTRIBUTION_CURRENT.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'checked_at':out['checked_at'],'proofs':proofs,'exact_original_parser':parser.PARSER_REVISION},indent=2)+'\n');t.replace(p);return out
if __name__=='__main__':
 r=summarize();print(json.dumps({'accepted':r['accepted_complete_APIs'],'rows':[(x['ordinal'],x['device_us']) for x in r['rows']]}))
