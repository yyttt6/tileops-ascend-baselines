"""Independent local CPU raw replay; same guard/source/output contract per exact key."""
import pathlib,json,hashlib,importlib.util,sys,types,statistics,fcntl,os,time
P=pathlib.Path(__file__).resolve().parent;R='/opt/baseline-over3-20261110/ordered100-20261112';M=P/'remote'
spec=importlib.util.spec_from_file_location('full32',P.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
pkg=types.ModuleType('tools');pkg.kernel_task_profile=parser;sys.modules['tools']=pkg;sys.modules['tools.kernel_task_profile']=parser
from restricted_profiler.partition import partition

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def path(s):return M/pathlib.Path(s).relative_to(R)
with (P/'math-bmm-transformer-raw-audit.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);old=json.loads((P/'MATH_BMM_TRANSFORMER_RAW_AUDIT.json').read_text()) if (P/'MATH_BMM_TRANSFORMER_RAW_AUDIT.json').exists() else {};out={};t=time.perf_counter()
 for cp in P.glob('rollout-math-bmm-transformer-alternative-*-h[12]-v1.CHECKPOINT.json'):
  label=cp.name.removesuffix('.CHECKPOINT.json');checkpoint_sha=sha(cp)
  if old.get(label,{}).get('checkpoint_sha256')==checkpoint_sha:out[label]=old[label];continue
  c=json.loads(cp.read_text());g=c['guard'];assert g['performance_valid'] and g['workload_exit_code']==0 and g['phase']=='finished'
  batch=c['batch'];B=M/'runs'/c['tag'];assert json.loads((B/'BATCH.json').read_text())==batch and sha(B/'SOURCE.json')==batch['source_sha256']
  rows=[]
  for r in batch['results']:
   i=r['ordinal'];assert json.loads((B/str(i)/'result.json').read_text())==r
   if r['status']!='raw_checked_pending_guard':rows.append({'ordinal':i,'status':r['status'],'error':r.get('error'),'phase':r.get('phase')});continue
   timing=r['timing'];session=timing.get('measurement_protocol')=='experimental-baseline-full32-marker-session-v1'
   if session:
    info=timing['single_session'];raw=path(info['path']);assert parser.trace_digest(raw)==info['sha256'];pieces,proof=partition(raw,15);proof['samples']=pieces
    assert proof==json.loads((B/str(i)/'session/partition.json').read_text())
   else:
    pieces=[]
    for prof in timing['retained_profiles']:
     raw=path(prof['profile_path']);assert parser.trace_digest(raw)==prof['profile_trace_sha256'];piece=parser.parse_profile(raw,1);assert piece['profile_named_task_coverage_complete'];pieces.append(piece)
   assert len(pieces)==15 and len(timing['latency_samples_ms'])==15
   assert all(abs(x['latency_ms']-v)<1e-9 for x,v in zip(pieces,timing['latency_samples_ms']))
   assert abs(statistics.mean(timing['latency_samples_ms'])-timing['latency_ms'])<1e-9
   assert r['check']['checked_outputs']==r['check']['total_outputs'] and r['check']['checked_outputs']>0 and not r['check']['unchecked_reason'] and r['inputs_preserved_by_original_verifier'] and len(r['wall_calls'])>=19
   if r.get('host_wall_formal_slice')==[4,19]:
    wall=r['wall_calls'][4:19];assert len(wall)==15 and all(v['ms']>0 and not v['child'] for v in wall)
    assert abs(sum(v['ms'] for v in wall)/15*1000-r['host_us_diagnostic'])<1e-6
   prep=path(r['prepare_metadata_path']) if r.get('prepare_metadata_path') else M/'prepared-rollout'/str(i)/'prepare.json'
   # Multiple arms use the same immutable prepare SHA; metadata must match.
   assert sha(prep)==r['prepare_sha256'];meta=json.loads(prep.read_text());assert meta['payload_sha256']==r['payload_sha256']
   rows.append({'ordinal':i,'status':'formal_raw_and_full_output_verified','windows':15,'protocol':timing.get('measurement_protocol','original-isolated'),'result_sha256':sha(B/str(i)/'result.json'),'device_us':r['device_us'],'check':r['check'],'metric':r['metric']})
  out[label]={'checkpoint_sha256':checkpoint_sha,'guard_valid':True,'runtime':batch['runtime'],'rows':rows,'guarded_seconds':c['guarded_seconds']}
 tmp=P/'MATH_BMM_TRANSFORMER_RAW_AUDIT.json.tmp';tmp.write_text(json.dumps(out,indent=2));os.replace(tmp,P/'MATH_BMM_TRANSFORMER_RAW_AUDIT.json')
print('AUDITED',len(out),'batches',sum(sum(r['status']=='formal_raw_and_full_output_verified' for r in b['rows']) for b in out.values()),'valid legs',round(time.perf_counter()-t,3),'seconds')
