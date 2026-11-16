#!/usr/bin/env python3
"""Read receipts and accept only complete original checks plus owned native launches."""
import argparse,hashlib,json,re,shutil
from datetime import datetime
from bisect import bisect_left
from pathlib import Path
import sys
sys.path.insert(0,'/root/dyq/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math')
from specs import ROOT,CACHE,SOURCE,EVIDENCE,SPECS
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
guardfile=run/'guard_exit.json'
guard_ok=guardfile.exists() and json.loads(guardfile.read_text())['exit_code']==0
events=[json.loads(x) for x in (run/'native-events.jsonl').read_text().splitlines()]
graph_calls=[json.loads(x) for x in (run/'graph-calls.jsonl').read_text().splitlines()] if (run/'graph-calls.jsonl').exists() else []
child_records=[(f,json.loads(f.read_text())) for f in (run/'native-children').glob('*.json')]
def event_time(e):
 m=re.search(r'(\d{4}-\d\d-\d\d)-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',e['text'])
 return datetime.fromisoformat(f'{m[1]}T{m[2]}.{m[3]}{m[4]}+00:00') if m else datetime.fromisoformat(e['captured_at'])
def file_record(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
index_path=EVIDENCE/'native_provenance_index.json'
index=json.loads(index_path.read_text()) if index_path.exists() else {}
# A failed verifier does not erase the origin of a compiled binary. Reuse
# only exact archived artifacts plus an observed own compile/launch; a new
# case must still pass its entire unchanged check and current native launch.
for prior in (EVIDENCE/'runs').glob('*/acceptance.json'):
 for row in json.loads(prior.read_text()).get('cases',[]):
  receipt=ROOT/row['receipt']
  if not receipt.exists():continue
  prior_receipt=json.loads(receipt.read_text())
  artifact_hashes=[a['sha256'] for a in row.get('native_artifacts',[]) if (ROOT/a['path']).is_file() and hashlib.sha256((ROOT/a['path']).read_bytes()).hexdigest()==a['sha256']]
  if not artifact_hashes:continue
  for pf in row.get('own_native_proof',[]):
   compile=pf.get('compile',{});match=re.search(r'kernel\[([^\]]+)\]',compile.get('text',''))
   if match and '/allpairs_math_math/' in compile.get('text','') and match[1] in pf.get('launch',{}).get('text',''):
    index.setdefault(match[1],{'compile':compile,'artifact_sha256':artifact_hashes,'native_sources':{k:v for k,v in prior_receipt.get('source_sha256',{}).items() if k.startswith('ascend_baselines/ops-math/')},'origin_acceptance':str(prior.relative_to(ROOT)),'numerical_result_not_inherited':True})
rows=[]
for f in sorted(run.glob('[0-9]*.json')):
 r=json.loads(f.read_text());start=datetime.fromisoformat(r['started_at']);end=datetime.fromisoformat(r['finished_at'])
 interval=[e for e in events if start<=event_time(e)<=end]
 own=[e for e in interval if 'single op compile success' in e['text'] and '/allpairs_math_math/' in e['text']]
 kernels=[];proof=[]
 for event in own:
  m=re.search(r'kernel\[([^\]]+)\]',event['text'])
  if m:
   k=m[1];launch=[e for e in interval if 'StreamLaunchKernelV' in e['text'] and k in e['text']]
   if launch:kernels.append(k);proof.append({'compile':event,'launch':launch[0]})
 # Graph compiler's cache still has an independently attributed prior compile.
 for event in events:
  if 'single op compile success' not in event['text'] or '/allpairs_math_math/' not in event['text']:continue
  m=re.search(r'kernel\[([^\]]+)\]',event['text'])
  if m and m[1] not in kernels:
   launch=[e for e in interval if 'StreamLaunchKernelV' in e['text'] and m[1] in e['text']]
   if launch:kernels.append(m[1]);proof.append({'compile':event,'launch':launch[0]})
 native_hashes={k:v for k,v in r.get('source_sha256',{}).items() if k.startswith('ascend_baselines/ops-math/')}
 for kernel,previous in index.items():
  if kernel in kernels:continue
  launch=[e for e in interval if 'StreamLaunchKernelV' in e['text'] and kernel in e['text']]
  previous_sources=previous.get('native_sources',{})
  shared=set(previous_sources)&set(native_hashes)
  if launch and any('/op_kernel/' in k for k in shared) and all(previous_sources[k]==native_hashes[k] for k in shared):
   cached=list((CACHE/'runtime-cache/kernel_cache').rglob(kernel+'.o'))
   if any(hashlib.sha256(q.read_bytes()).hexdigest() in previous['artifact_sha256'] for q in cached):
    kernels.append(kernel);proof.append({'compile':previous['compile'],'launch':launch[0],'cached_binary_hash_matched':True})
 artifacts=[]
 for kernel in kernels:
  for artifact in (CACHE/'runtime-cache/kernel_cache').rglob(kernel+'.*'):
   if artifact.suffix in ('.json','.o'):
    dest=run/'native-artifacts'/artifact.name;dest.parent.mkdir(exist_ok=True)
    if not dest.exists():shutil.copy2(artifact,dest)
    artifacts.append(file_record(dest))
 v=r.get('verification',{});complete=bool(v.get('checked_outputs')) and v.get('checked_outputs')==v.get('total_outputs')
 required_graphs={c['graph'] for c in graph_calls if start<=datetime.fromisoformat(c['started_at'])<=datetime.fromisoformat(c['finished_at'])<=end}
 attributed_graphs=set()
 for pf in proof:
  match=re.search(r'opType\[([^\]]+)\]',pf['compile']['text'])
  if match:attributed_graphs.add(match[1])
 missing_graphs=sorted(required_graphs-attributed_graphs)
 graph_for_kernel={}
 for kernel,pf in zip(kernels,proof):
  match=re.search(r'opType\[([^\]]+)\]',pf['compile']['text'])
  if match:graph_for_kernel[kernel]=match[1]
 launches_by_graph={}
 for event in interval:
  if 'StreamLaunchKernelV' not in event['text']:continue
  for kernel,graph_name in graph_for_kernel.items():
   if kernel in event['text']:
    launches_by_graph.setdefault(graph_name,[]).append((event_time(event),kernel,event));break
 for launches in launches_by_graph.values():launches.sort(key=lambda x:x[0])
 launch_times={name:[x[0] for x in launches] for name,launches in launches_by_graph.items()}
 graph_call_proofs=[];missing_graph_calls=[]
 for call in graph_calls:
  call_start=datetime.fromisoformat(call['started_at']);call_end=datetime.fromisoformat(call['finished_at'])
  if not start<=call_start<=call_end<=end:continue
  graph_name=call['graph'];position=bisect_left(launch_times.get(graph_name,[]),call_start)
  launches=launches_by_graph.get(graph_name,[])
  if position<len(launches) and launches[position][0]<=call_end:
   when,kernel,event=launches[position]
   graph_call_proofs.append({'graph':graph_name,'started_at':call['started_at'],'finished_at':call['finished_at'],'kernel':kernel,'launch_file':event['file'],'launch_line':event['line']})
  else:missing_graph_calls.append(call)
 requires_layernorm=any(name.endswith('/layernorm/libops_math_layernorm.so') for name in r.get('source_sha256',{}))
 # Graph proofs do not prove the separately compiled fast-launch example.
 # Extend this only after inspecting actual binary registration/launch events.
 layernorm_proof=[]
 if requires_layernorm:
  binary=CACHE/'layernorm/libops_math_layernorm.so';compiled=CACHE/'layernorm/receipt.json'
  source=SOURCE/'examples/fast_kernel_launch_example/csrc/layer_norm/dav-2201/layer_norm.asc'
  binary_key=str(binary.relative_to(ROOT));build=json.loads(compiled.read_text())
  binary_hash=hashlib.sha256(binary.read_bytes()).hexdigest();source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
  matched=(build.get('compile_exit_code')==0 and build.get('link_exit_code')==0 and build.get('binary_sha256')==binary_hash and build.get('source_sha256')==source_hash and r.get('source_sha256',{}).get(binary_key)==binary_hash)
  symbol='_ZN10ascend_ops9LayerNorm15layer_norm_fp32EPhS1_S1_S1_llllf'
  if matched:
   for childfile,child in child_records:
    if child.get('op')!=r['op'] or child.get('case_id')!=r['case_id'] or child.get('status')!='native_execution_complete' or child.get('source_sha256',{}).get(binary_key)!=binary_hash:continue
    child_start=datetime.fromisoformat(child['started_at']);child_end=datetime.fromisoformat(child['finished_at'])
    for event in interval:
     if 'StreamLaunchKernelV' not in event['text'] or symbol not in event['text'] or f"RUNTIME({child['pid']}," not in event['text']:continue
     if not start<=child_start<=event_time(event)<=child_end<=end:continue
     dest=run/'native-artifacts'/f'libops_math_layernorm-{binary_hash[:16]}.so';dest.parent.mkdir(exist_ok=True)
     if not dest.exists():shutil.copy2(binary,dest)
     if hashlib.sha256(dest.read_bytes()).hexdigest()!=binary_hash:continue
     artifact=file_record(dest)
     if artifact not in artifacts:artifacts.append(artifact)
     layernorm_proof.append({'binary':artifact,'source':file_record(source),'compile_receipt':file_record(compiled),'child_record':file_record(childfile),'launch':event,'kernel_symbol':symbol,'binary_source_receipt_match':True})
 missing_native_examples=['layer_norm_fp32'] if requires_layernorm and not layernorm_proof else []
 child_proofs=[]
 for childfile,child in child_records:
  if child.get('op')!=r['op'] or child.get('case_id')!=r['case_id'] or child.get('status')!='native_execution_complete':continue
  if not start<=datetime.fromisoformat(child['started_at'])<=datetime.fromisoformat(child['finished_at'])<=end:continue
  child_sources=child.get('source_sha256',{})
  parent_sources=r.get('source_sha256',{})
  extra_sources={k:v for k,v in child_sources.items() if k not in parent_sources}
  shared_sources_match=all(parent_sources[k]==v for k,v in child_sources.items() if k in parent_sources)
  auxiliary_sources_audited=False
  allowed_meanpool_auxiliary={'ascend_baselines/ops-math/math/reduce_mean/framework/reduce_mean_onnx_plugin.cpp': '4bf9c7f61754655bb205772ed49e6397ef8cb6745b66d7a62b3b51da7c4db54f', 'ascend_baselines/ops-math/math/reduce_mean/framework/reduce_mean_tf_plugin.cpp': '519a1556733854fc0ad5409722dbd6b5672ccf7a25d85e12d747282b0f30e7aa', 'ascend_baselines/ops-math/math/reduce_mean/examples/test_geir_reduce_mean_with_cast.cpp': 'd6fdb0b0cbf042da3cf7fc2e196e2ca46908ee9ca826e41a67afba9bdfbd216e', 'ascend_baselines/ops-math/math/reduce_mean/examples/test_aclnn_mean_v2.cpp': '4d5377a7c1027a096b09b778cae938d33860767f6f4766bf7370450afbde407c', 'ascend_baselines/ops-math/math/reduce_mean/examples/test_geir_reduce_mean.cpp': 'f47e121326ecf96839dddee915c91323fc2ff3991ad71ee3a698ac0939276d43', 'ascend_baselines/ops-math/math/reduce_mean/examples/test_aclnn_global_average_pool.cpp': '3c03ca4e06b867853a0c7bb9d30696e6cbebed1c3d78f7bf0bc8cee698fa8739', 'ascend_baselines/ops-math/math/reduce_mean/examples/test_aclnn_mean.cpp': '1d5d7ab663a7269159a35f73d005affec1b472bf7e08bea2f3a51b9a819a604e', 'ascend_baselines/ops-math/math/reduce_mean/op_kernel_aicpu/reduce_mean_aicpu.h': '1e018e044abade9d55b801e5cab17f14093b26847d0384a7d9cc09fe4fc520d6', 'ascend_baselines/ops-math/math/reduce_mean/op_kernel_aicpu/reduce_mean_aicpu_def.cpp': '5d172d05608b72a64e2f4bbdec3cdcc5756bf4e110f4cf8c0807796a6318e346', 'ascend_baselines/ops-math/math/reduce_mean/op_kernel_aicpu/reduce_mean_aicpu.cpp': '50bb7d289f5972e8e0d1c50a785263e24915276e18bf4dbacb9935afce3f8edc', 'ascend_baselines/ops-math/math/reduce_mean/op_kernel_aicpu/reduce_op.h': 'b519d59eaf4545fdec298bfe8b1e23153972fdf54ee01c4db588908614f0c87c'}
  meanpool_key='TileOPs/benchmarks/ascend_all_pairs/ops_math/mean_pooling.py'
  meanpool_hash='946298a8b6fe83b89a894e2da7999758983e2506bb7ad33e6c4174bf2bae1bbb'
  if r['op']=='MeanPoolingFwdOp' and child.get('api')=='ops-math::ReduceMean(original chunk packing)' and extra_sources==allowed_meanpool_auxiliary and parent_sources.get(meanpool_key)==meanpool_hash and child_sources.get(meanpool_key)==meanpool_hash:
   current_auxiliary_match=all((ROOT/k).is_file() and hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in extra_sources.items())
   # The extra framework/example/AiCPU files are recorded, but the observed
   # computation is exclusively the declared and owned AiCore ReduceMean.
   aicore_mean=required_graphs=={'ReduceMean'} and any('opType[ReduceMean]' in pf['compile']['text'] and '/op_impl/ai_core/' in pf['compile']['text'] for pf in proof)
   auxiliary_sources_audited=current_auxiliary_match and aicore_mean
  if child_sources and shared_sources_match and (not extra_sources or auxiliary_sources_audited):
   child_proofs.append({'record':file_record(childfile),'pid':child['pid'],'process_group':child['process_group'],'api':child['api'],'additional_nonexecuted_vendor_sources':extra_sources if auxiliary_sources_audited else {},'additional_sources_current_hashes_checked':auxiliary_sources_audited})
 isolated_ok='isolated foreground process' not in r.get('api','') or bool(child_proofs)
 # Frozen verified Abs predates graph-call instrumentation. Its unchanged
 # adapter has exactly one Abs API invocation per complete native child.
 # Require the exact frozen source and an owned Abs launch in EACH child.
 legacy_abs_single_api_proofs=[]
 if r['op']=='AbsFwdOp' and not required_graphs and child_proofs:
  frozen_key='TileOPs/benchmarks/ascend_all_pairs/ops_math/validated_abs/graph.py'
  frozen_hash='0410b3e4c92638e8d6915725ad6112a2fc3f8ebf254d35a29d83b1e91929445d'
  if r.get('source_sha256',{}).get(frozen_key)==frozen_hash and hashlib.sha256((ROOT/frozen_key).read_bytes()).hexdigest()==frozen_hash:
   abs_kernels={k for k,g in graph_for_kernel.items() if g=='Abs'}
   for cp in child_proofs:
    cr=json.loads((ROOT/cp['record']['path']).read_text())
    if cr.get('api')!='ops-math::Abs via aclopCompileAndExecute' or cr.get('source_sha256',{}).get(frozen_key)!=frozen_hash:continue
    lo=datetime.fromisoformat(cr['started_at']);hi=datetime.fromisoformat(cr['finished_at'])
    hits=[ev for ev in interval if 'StreamLaunchKernelV' in ev['text'] and f"RUNTIME({cr['pid']}," in ev['text'] and lo<=event_time(ev)<=hi and any(k in ev['text'] for k in abs_kernels)]
    unique={(ev['file'],ev['line']):ev for ev in hits}
    if len(unique)==1:
     legacy_abs_single_api_proofs.append({'frozen_adapter':file_record(ROOT/frozen_key),'child_record':cp['record'],'child_pid':cr['pid'],'expected_api_calls':1,'owned_abs_launches':list(unique.values())})
 legacy_abs_complete=bool(child_proofs) and len(legacy_abs_single_api_proofs)==len(child_proofs)
 layernorm_call_coverage=[]
 if requires_layernorm:
  params=r['case_snapshot']['parameters'];expected=1
  if r['op']=='GroupNormFwdOp' and (params.get('weight') is not None or params.get('bias') is not None):expected=params['num_groups']
  if r['op'] in ('InstanceNormFwdOp','BatchNormFwdOp') and (params.get('weight') is not None or params.get('bias') is not None):expected=params['x'][0][1]
  for cp in child_proofs:
   matching=[pf for pf in layernorm_proof if pf['child_record']['path']==cp['record']['path']]
   actual=len({(pf['launch']['file'],pf['launch']['line']) for pf in matching})
   layernorm_call_coverage.append({'child_record':cp['record'],'expected_api_calls':expected,'observed_owned_launches':actual,'complete':actual==expected})
 layernorm_calls_complete=not requires_layernorm or bool(layernorm_call_coverage) and all(c['complete'] for c in layernorm_call_coverage)
 changed_sources=[]
 for name,expected in r.get('source_sha256',{}).items():
  current=ROOT/name
  if not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest()!=expected:changed_sources.append(name)
 graph_trace_required=not (requires_layernorm and r['op'] in ('LayerNormFwdOp','GroupNormFwdOp','InstanceNormFwdOp','BatchNormFwdOp'))
 graph_trace_complete=bool(required_graphs) or not graph_trace_required or legacy_abs_complete
 accepted=graph_trace_complete and layernorm_calls_complete and guard_ok and r.get('status')=='verified' and complete and isolated_ok and (bool(proof) or bool(layernorm_proof)) and bool(artifacts) and not missing_graphs and not missing_graph_calls and not missing_native_examples and not changed_sources
 if accepted:
  for kernel,pf in zip(kernels,proof):
   index[kernel]={'compile':pf['compile'],'artifact_sha256':[q['sha256'] for q in artifacts],'native_sources':{k:v for k,v in r.get('source_sha256',{}).items() if k.startswith('ascend_baselines/ops-math/')}}
 rows.append({'op':r['op'],'case_id':r['case_id'],'receipt':str(f.relative_to(ROOT)),'accepted':accepted,'complete_manifest_check':complete,'original_input_preservation':r.get('status')=='verified' and complete,'own_native_proof':proof,'native_artifacts':artifacts,'required_graphs':sorted(required_graphs),'attributed_graphs':sorted(attributed_graphs),'missing_graphs':missing_graphs,'missing_native_examples':missing_native_examples,'layernorm_proof':layernorm_proof,'isolated_native_child_proofs':child_proofs,'changed_sources':changed_sources,'reason':None if accepted else r.get('error') or 'Missing complete numerical, current source, or owned-native evidence'})
 rows[-1].update(graph_call_proofs=graph_call_proofs,missing_graph_calls=missing_graph_calls,graph_trace_required=graph_trace_required,graph_trace_complete=graph_trace_complete,legacy_abs_single_api_proofs=legacy_abs_single_api_proofs,layernorm_call_coverage=layernorm_call_coverage,layernorm_calls_complete=layernorm_calls_complete)
index_path.write_text(json.dumps(index,indent=2))
(run/'acceptance.json').write_text(json.dumps({'schema':'ops-math-native-acceptance/v1','auditor':file_record(Path(__file__).resolve()),'events':file_record(run/'native-events.jsonl'),'cases':rows},indent=2))
print(json.dumps({'accepted':sum(r['accepted'] for r in rows),'cases':len(rows)}))

# Publication is case-scoped and happens after coverage refresh in publish.py.
