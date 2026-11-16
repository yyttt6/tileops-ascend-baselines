"""Foreground continuation in frozen order; bounded processes, immutable evidence."""
import pathlib,sys,json,time,subprocess,fcntl,datetime,shlex,hashlib
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import rollout_controller as c
rows=json.loads((P/'TASKS.json').read_text());assert len(rows)==2119
assert json.loads((P/'RESULT.json').read_text())['gate']['passed']
lock=(P/'continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ledger_path=P/'ROLLOUT_LEDGER.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
blocks={};library_blocks={'ops-transformer':{'reason':'CPU registry import repaired; original native OPP and infer shim absent; source-bound native profile required before further device trials','evidence':'rollout-766-766.CHECKPOINT.json'}};start=time.monotonic()
def put(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=str));tmp.replace(p)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def profile(lib):
 if lib=='sgl-kernel-npu':return 'r-sgl'
 if lib=='tilelang':return 'r-tilelang-original'
 if lib=='tilelang-ascend':return 'r-tla-original'
 if lib=='vllm-ascend':return 'r-vllm'
 if lib in ('catlass','flash-linear-attention-npu','ops-math','ops-nn','ops-tensor','ops-transformer'):return 'r-sdk91'
 return 'r-template'
def publish(status,ids=()):
 valid=sum(e.get('result',{}).get('status')=='raw_checked_pending_guard' for e in ledger.values())
 put(P/'CONTINUATION.json',{'status':status,'updated_at':stamp(),'indices':list(ids),'recorded':len(ledger),'new_valid':valid,'pending_or_failed':len(ledger)-valid,'remaining_unvisited':len(rows)-100-len(ledger),'first100_closed':True,'first100_matched_speedup':6.8984992578,'elapsed_controller_seconds':time.monotonic()-start,'library_blocks':{str(k):v for k,v in blocks.items()},'history_or_public_ratios_updated':False,'full_api_adjustment_review':'fresh measured originals remain pending alternative-API review; not automatically counted as improvements'})
def save(event):
 for r in event['batch']['results']:
  existing=ledger.get(str(r['ordinal']),{})
  if existing.get('result',{}).get('status')=='raw_checked_pending_guard':continue
  ledger[str(r['ordinal'])]={'label':event['label'],'tag':event['tag'],'checkpoint_sha256':c.sha(P/(event['label']+'.CHECKPOINT.json')),'result':{k:v for k,v in r.items() if k not in ('timing','edge_probes','wall_calls')},'full_result_evidence':str(pathlib.Path('remote/runs')/event['tag']/str(r['ordinal'])/'result.json')}
 put(ledger_path,ledger)
# Adopt completed qualified batches on resume; no duplicate device execution.
superseded=set(json.loads((P/'PINNED_DEPENDENCY_SUPERSEDED.json').read_text())['checkpoint_labels'])
committed_labels={e.get('label') for e in ledger.values() if e.get('label')}
for cp in sorted(P.glob('rollout-*.CHECKPOINT.json')):
 if cp.name.removesuffix('.CHECKPOINT.json') in committed_labels:continue
 e=json.loads(cp.read_text())
 if e['label'] not in superseded and not e['label'].startswith(('rollout-fla-candidate-','rollout-sgl-wrapper-candidate-','rollout-vllm-norm-candidate-','rollout-tla-unpack-candidate-','rollout-tla-zero-pad-candidate-','rollout-tla-abs-unary-candidate-','rollout-vllm-rms-api-candidate-','rollout-sgl-dequant-api-candidate-','rollout-sgl-dequant-channel-candidate-','rollout-sgl-dequant-block-candidate-','rollout-sgl-lerp-tensor-candidate-')) and all(r['ordinal']>=100 for r in e['batch']['results']):save(e)

candidate_path=P/'ROLLOUT_CANDIDATES.json';candidates=json.loads(candidate_path.read_text()) if candidate_path.exists() else {}
def candidate_hook(ids):
 eligible=[j for j in ids if rows[j]['library']=='flash-linear-attention-npu' and rows[j]['op']=='AddFwdOp' and str(j) not in candidates and ledger.get(str(j),{}).get('result',{}).get('status')=='raw_checked_pending_guard']
 if not eligible:return
 label='rollout-fla-candidate-'+str(eligible[0])+'-'+str(eligible[-1])
 while True:
  with (P/'controller.lock').open('a') as run_lock:
   fcntl.flock(run_lock,fcntl.LOCK_EX);event=c.run(profile('flash-linear-attention-npu'),'fla-candidate',eligible,label)
  if event is None:publish('waiting_ownhost_for_reviewed_candidate',eligible);time.sleep(30);continue
  break
 if 'batch' not in event:raise RuntimeError('review candidate wholeprocess/guard invalid; retain evidence before continuation')
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
 for r in event['batch']['results']:
  j=r['ordinal'];original=ledger[str(j)]['result'];record={'label':label,'result':r,'selection':'original','original_evidence':ledger[str(j)]['label']+'.CHECKPOINT.json'}
  if r['status']=='raw_checked_pending_guard':
   assert r['prepare_sha256']==original['prepare_sha256'] and r['payload_sha256']==original['payload_sha256']
   record['device_reduction_percent']=100*(1-r['device_us']/original['device_us'])
   record['selection']='zero_pad_wrapper' if r['device_us']<original['device_us'] else 'original'
  candidates[str(j)]=record
 put(candidate_path,candidates)
 # A numerical failure is kept, and the unstarted remainder is handled separately.
 done={r['ordinal'] for r in event['batch']['results']}
 unstarted=[j for j in eligible if j not in done]
 if unstarted:candidate_hook(unstarted)

def wait_global_disk(ids):
 while True:
  if (P/'STOP').exists() or (P.parent/'STOP').exists():raise RuntimeError('STOP while waiting disk')
  code="import shutil,json;print(json.dumps([shutil.disk_usage('/').free,shutil.disk_usage('/workspace/user_data').free]))"
  free=json.loads(c.remote('python3 -c '+shlex.quote(code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  if min(free)>=20_000_000_000:return
  publish('waiting_global_disk_floor_no_cases_failed',ids)
  put(P/'DISK_WAIT.json',{'free_bytes':free,'threshold_bytes':20_000_000_000,'action':'global resource backoff; do not classify individual providers or markcases failed','updated_at':stamp()})
  time.sleep(30)

route_catalog=json.loads((P/'CPU_TLA_ROUTE_CATALOG.json').read_text())
assert route_catalog['NoNPU'] and route_catalog['classic_lookup_AST_proved_op_only']
for name,digest in route_catalog['sources'].items():
 assert c.sha(P/name)==digest, 'CPU route source changed: '+name
route_ops={lib:set(data['available_ops']) for lib,data in route_catalog['libraries'].items()}
i=100
while i<len(rows):
 if (P/'STOP').exists() or (P.parent/'STOP').exists():publish('stopped_by_user');sys.exit(0)
 if str(i) in ledger:
  candidate_hook([i]);i+=1;continue
 row=rows[i];group=(row['library'],row['op'])
 if row['library'] in route_ops and row['op'] not in route_ops[row['library']]:
  ledger[str(i)]={'result':{'ordinal':i,'key':[row['library'],row['op'],row['case']],'status':'provider_registry_route_missing','reason':'CPU-lazy package and original classic fallback have no registered provider for this operator; nativeAPI absence not inferred','evidence':'CPU_TLA_ROUTE_CATALOG.json'}}
  put(ledger_path,ledger);publish('CPU_missing_registry_route_no_device', [i]);i+=1;continue
 if row['library'] in library_blocks:
  ledger[str(i)]={'result':{'ordinal':i,'key':[row['library'],row['op'],row['case']],'status':'provider_import_dependency_pending','cause':library_blocks[row['library']]}};put(ledger_path,ledger);i+=1;continue
 if row['library']=='flash-linear-attention-npu' and (row['op']=='BmmFwdOp' or (row['op']=='GemmFwdOp' and 'bfloat16' in row['case'])):
  ledger[str(i)]={'result':{'ordinal':i,'key':[row['library'],row['op'],row['case']],'status':'provider_deployment_pending','reason':'original native FLA package absent; source import fixed but no unverifiable binary substituted','evidence':'rollout-150-157.CHECKPOINT.json'}};put(ledger_path,ledger);i+=1;continue
 if group in blocks:
  ledger[str(i)]={'result':{'ordinal':i,'key':[row['library'],row['op'],row['case']],'status':'provider_dependency_deferred','cause':blocks[group]}};put(ledger_path,ledger);i+=1;continue
 ids=[]
 batch_limit=1 if row['library'] in ('ops-nn','ops-math','ops-transformer','ops-tensor') or (row['library']=='tilelang-ascend' and row['op'] in ('AmaxFwdOp','AminFwdOp')) else 4
 for j in range(i,min(i+batch_limit,len(rows))):
  r=rows[j]
  if (r['library'],r['op'])!=group or str(j) in ledger:break
  if row['library']=='flash-linear-attention-npu' and (r['op']=='BmmFwdOp' or (r['op']=='GemmFwdOp' and 'bfloat16' in r['case'])):break
  ids.append(j)
 wait_global_disk(ids)
 family=profile(row['library']);label='rollout-'+str(i)+'-'+str(ids[-1])+('-dep2' if 122<=i<=149 else '')+('-metadata2' if row['library']=='ops-nn' else '')+('-bounded1' if row['library'] in ('ops-nn','ops-math','ops-transformer','ops-tensor') else '')+('-pathfix2' if row['library']=='sgl-kernel-npu' and row['op']=='CountNonzeroFwdOp' else '')+('-layoutasset2' if row['library']=='sgl-kernel-npu' and row['op']=='SSDChunkCouplingFwdOp' else '')+('-resource-single1' if row['library']=='tilelang-ascend' and row['op'] in ('AmaxFwdOp','AminFwdOp') else '-cache-evidence2' if row['library']=='tilelang-ascend' else '');publish('cpu_preparation',ids)
 prep_tag=label+'-prepare-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f')
 cmd='bash '+c.R+'/launch.sh '+family+' prepare '+prep_tag+' '+' '.join(map(str,ids))
 t=time.monotonic()
 with (P/(prep_tag+'.log')).open('w') as out:rc=c.remote(cmd,stdout=out,stderr=subprocess.STDOUT).returncode
 with (P/'CPU_PREP_TIMING.jsonl').open('a') as out:out.write(json.dumps({'ids':ids,'seconds':time.monotonic()-t,'exit_code':rc,'log':prep_tag+'.log'})+'\n')
 if rc==0:
  proof_code='import pathlib,json;p=pathlib.Path('+repr(c.R)+');ids='+repr(ids)+';print(json.dumps({i:[name for name in ["prepare.json","case.pkl","payload.pt"] if not (p/"prepared-rollout"/str(i)/name).is_file()] for i in ids}))'
  artifacts=json.loads(c.remote('python3 -c '+shlex.quote(proof_code),stdout=subprocess.PIPE,text=True,check=True).stdout)
  put(P/(prep_tag+'.ARTIFACTS.json'),artifacts)
  if any(artifacts.values()):
   publish('CPU_prepare_zero_exit_missing_outputs_no_device_submission',ids)
   raise RuntimeError('CPU preparation returned0 but required outputs absent; intervention required: '+prep_tag)
 if rc and 'disk20e9 floor' in (P/(prep_tag+'.log')).read_text():
  wait_global_disk(ids);continue
 if rc and 'CPU_REGISTRY_DEPENDENCY_MISSING:' in (P/(prep_tag+'.log')).read_text():
  library_blocks[row['library']]={'reason':'CPU registry import metadata missing; library-level hold','log':prep_tag+'.log'};publish('library_import_dependency_pending',ids);continue
 if rc and 'FROZEN_CASE_UNAVAILABLE_JSON=' in (P/(prep_tag+'.log')).read_text():
  line=next(s for s in (P/(prep_tag+'.log')).read_text().splitlines() if s.startswith('FROZEN_CASE_UNAVAILABLE_JSON='))
  missing=json.loads(line.split('=',1)[1]);assert missing
  for bad in missing:
   assert (rows[bad]['library'],rows[bad]['op'])==group
   if str(bad) not in ledger:
    ledger[str(bad)]={'result':{'ordinal':bad,'key':[rows[bad]['library'],rows[bad]['op'],rows[bad]['case']],'status':'provider_dependency_deferred','reason':'Exact frozen case absent from factory enumeration, no alias/substitution','phase':'CPU_prepare_exact_case','log':prep_tag+'.log','no_device_attempt':True}}
  put(ledger_path,ledger);publish('exact_case_gap_preserved_other_cases_continue',ids);continue
 if rc:
  reason={'phase':'CPU_prepare','exit_code':rc,'log':prep_tag+'.log','no_device_attempt':True};blocks[group]=reason;publish('provider_preparation_deferred',ids);continue
 while True:
  with (P/'controller.lock').open('a') as run_lock:
   fcntl.flock(run_lock,fcntl.LOCK_EX);event=c.run(family,'auto',ids,label)
  if event is None:publish('waiting_for_ownhost_idle',ids);time.sleep(30);continue
  break
 if 'batch' not in event:
  publish('process_or_guard_failure_requires_review',ids)
  # A child may have produced correctness files, but the entire guard failed.
  # Do not accept any performance or automatically rerun these cases.
  for j in ids:ledger[str(j)]={'result':{'ordinal':j,'key':[rows[j]['library'],rows[j]['op'],rows[j]['case']],'status':'guard_or_process_invalid_pending','evidence':label+'.FAILED.json'}}
  put(ledger_path,ledger);blocks[group]={'phase':'whole_process_failure','evidence':label+'.FAILED.json'};continue
 subprocess.run([sys.executable,str(P/'audit_rollout.py')],check=True)
 save(event)
 result=event['batch']['results'][-1]
 if result['status']!='raw_checked_pending_guard':
  error=result.get('error','')
  # A common missing deployment is reviewed once, not rediscovered on every case.
  if result.get('phase') in ('load','factory') and any(s in error for s in ['ModuleNotFoundError','ImportError','FileNotFoundError','undefined symbol','No such file','cannot open shared object','deployment not available','deployed Python/source mismatch','SGL source mismatch','is not in the subpath']):blocks[group]={'phase':result['phase'],'error':error,'evidence':label+'.CHECKPOINT.json'}
 if result['status']!='raw_checked_pending_guard':
  error=result.get('error','')
  if any(s in error for s in ['Tensor out not implemented for DT_UINT8', "Select this library's OPP with baseline-run", 'Load the isolated original vendor before importing Torch.', 'Unresolved call Op(tl.', 'Live foreground guard identity mismatch','Live foreground npu-guard required','Guard kernel lock is not held','Verified Python ABI/runtime required','Wrong loaded compiler','Frozen dependency changed:','FileNotFoundError','ModuleNotFoundError','Missing library:','Installed tilelang differs from selected source','requires its own runtime']):
   blocks[group]={'phase':'pinned_guard_or_environment_dependency','error':error,'evidence':label+'.CHECKPOINT.json','no_blind_same_dependency_retries':True}
 candidate_hook([r['ordinal'] for r in event['batch']['results']])
 publish('batch_audited',ids)
 # The worker stops after one case failure. Unstarted cases follow in a new process.
 i+=1
publish('all_ordered_rows_traversed_pending_adjustment_and_failure_closure')
