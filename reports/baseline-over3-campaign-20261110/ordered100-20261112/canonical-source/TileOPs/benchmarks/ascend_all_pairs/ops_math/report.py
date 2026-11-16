#!/usr/bin/env python3
"""CPU-only coverage and evidence inventory; never equate numerical pass with ownership."""
import hashlib,json,re
from collections import Counter
from functools import lru_cache
from datetime import datetime,timezone
from pathlib import Path
from specs import ROOT,SOURCE,CACHE,EVIDENCE,SPECS,files
from semantic_limits import reason as semantic_reason
from composition_specs import COMPOSITIONS

@lru_cache(maxsize=None)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def evidence(p,line=1):return {'file':str(p.relative_to(ROOT)),'line':line,'sha256':sha(p)}
inv=json.loads((EVIDENCE.parent/'inventory.json').read_text())
blocker_file=EVIDENCE/'build_guard_001.json'
blocker=json.loads(blocker_file.read_text()) if blocker_file.exists() else None
build_receipt=CACHE/'build_receipt.json'
build=json.loads(build_receipt.read_text()) if build_receipt.exists() else {}
blocker_active=bool(blocker and blocker.get('exit_code') in (75,76,78) and build.get('finished_at','')<blocker['finished_at'])
# The complete source path catalog is evidence for negative searches, including examples.
catalog=[str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file() and not any(s in ('build','build_out','.git','third_party') for s in p.relative_to(SOURCE).parts)]
(EVIDENCE/'source_catalog.json').write_text(json.dumps({'root':str(SOURCE),'files':sorted(catalog)},indent=2))
cat=evidence(EVIDENCE/'source_catalog.json')
reasons={
 'attention':'No owned full attention kernel/API: arithmetic primitives and transform_bias_rescale_qkv do not implement attention softmax/cache/sequence contracts.',
 'linear_attention':'No complete audited own-library route for the stateful delta/GLA/GDN/KDA recurrence, all chunk outputs and backward contracts; no matching native recurrence API was found.',
 'gemm':'The implemented original weighted-reduction route covers selected forward dot products; this remaining full matrix/output contract has no audited adapter.',
 'conv_pool':'No owned convolution or max/average pooling API; reshape/im2col is not convolution. MeanPooling is separately audited as ReduceMean.',
 'normalization':'No complete audited adapter for this remaining normalization/output contract. Selected forward LayerNorm and native primitive compositions are handled separately.',
 'quantization':'No complete manifest scale/round/clamp/packing quantization or dequantization implementation; cast/ada_cast do not provide that contract.',
 'moe':'No owned expert MLP/router/permutation API with required routing metadata. DynamicPartition alone lacks counts/prefix/inverse outputs.',
 'rope':'Only ONNX NpuRotaryMul plugin declarations; no owned rotary device kernel. Layout alone cannot generate rotary coefficients and rotate.',
 'ssm':'No owned Mamba/SSD/selective-scan recurrence and state contract; Cumsum is a different operator.',
 'sampling':'TopKV2 returns selected values/indices, not tie-preserving per-row logit masks or categorical/speculative draws; random generators alone are insufficient.',
 'other':'No equivalent owned kernel for the manifest sequence/transform contract.',
 'elementwise':'No complete audited own-library adapter for this remaining activation/gated/generative output contract. Native primitive compositions are permitted and selected implemented routes are classified separately.'}
specific={
 'CumprodFwdOp':('dependency_only','math/cumprod/op_api/cumprod.cpp','The complete public Cumprod implementation calls ADD_TO_LAUNCHER_LIST_AICPU; this checkout supplies no own Cumprod device kernel.'),
 'FFTC2CFwdOp':('dependency_only','math/stft/op_host/op_api/stft.cpp','Complex STFT path delegates AiCPU; RFFT1D is real-to-complex and does not satisfy complete C2C/inverse/norm contract.'),
 'CountNonzeroFwdOp':('unsupported_math',None,'No owned count_nonzero kernel/API; ReduceAll/Any return booleans and ReduceSum does not test nonzero.'),
 'VectorNormFwdOp':('unsupported_math',None,'No owned vector-norm kernel/API; ReduceSum/Max alone lack abs/power and final-root semantics.'),
 
 
 'TopKSelectFwdOp':('unsupported_math','math/top_k_v2/op_host/top_k_v2_def.cpp','TopKV2 lacks per-row starts/ends mask contract; generic TopK does not implement attention-indexer selection.'),
 'GeluFwdOp':('unsupported_math','examples/fast_kernel_launch_example/csrc/gelu/dav-2201/gelu.asc','Owned example implements only tanh approximate Gelu; manifest requires approximate=none, so source mathematics do not match.'),
 'GeluTanhAndMulFwdOp':('unsupported_math','examples/fast_kernel_launch_example/csrc/gelu/dav-2201/gelu.asc','Owned example provides tanh Gelu only, without required gated multiplication.'),
}
receipts={}
for f in sorted((EVIDENCE/'runs').glob('*/*.json')):
 try:r=json.loads(f.read_text())
 except Exception:continue
 if isinstance(r,dict) and 'op'in r and 'case_id'in r:receipts.setdefault((r['op'],r['case_id']),[]).append((f,r))
for history in receipts.values():history.sort(key=lambda pair:pair[1].get('finished_at',pair[1].get('started_at','')))
acceptances={}
for f in (EVIDENCE/'runs').glob('*/acceptance.json'):
 for r in json.loads(f.read_text()).get('cases',[]):
  history=receipts.get((r['op'],r['case_id']),[])
  if history and r.get('receipt')==str(history[-1][0].relative_to(ROOT)):acceptances[(r['op'],r['case_id'])]=(f,r)
historical={}
hf=EVIDENCE/'historical_add_revalidation.json'
if hf.exists():
 for r in json.loads(hf.read_text())['cases']:
  if r['accepted']:historical[(r['op'],r['case_id'])]=r
rows=[]
for info in inv['operators']:
 op=info['op']; source=[]; reason=None
 if op in SPECS:
  root=SOURCE/SPECS[op]['directory']; defs=list((root/'op_host').glob('*def*'))
  for f in defs:source.append(evidence(f,next((i for i,s in enumerate(f.read_text().splitlines(),1) if 'class '+SPECS[op]['graph'] in s),1)))
  status='supported_untested';reason='Original native graph API mapped; all manifest cases require fresh numerical and ownership verification.'
 elif op=='AddFwdOp':
  status='supported_untested';reason='Existing native ACLNN Add route is executable; original full output and signed-zero checks preserved.';source=[evidence(SOURCE/'math/add/op_api/aclnn_add.cpp')]
 elif op in ('LayerNormFwdOp','AdaLayerNormFwdOp','GroupNormFwdOp','InstanceNormFwdOp','BatchNormFwdOp','MeanPoolingFwdOp','StdFwdOp','VarFwdOp','VarMeanFwdOp'):
  status='supported_untested';reason='Owned source route under deployment/adapter preparation.'
  f=SOURCE/('examples/fast_kernel_launch_example/csrc/layer_norm/dav-2201/layer_norm.asc' if 'Norm' in op else 'math/reduce_mean/op_host/reduce_mean_def.cpp' if op=='MeanPoolingFwdOp' else 'math/reduce_var/op_host/reduce_var_def.cpp')
  source=[evidence(f)]
  if op=='BatchNormFwdOp':
   reason='Training without stored statistics uses the original LayerNorm example. Tracked/eval uses original native ReduceVar/ReduceMean and arithmetic, returning all three outputs without mutating inputs; device validation pending.'
   source.append(evidence(Path(__file__).with_name('running_stats.py')))
 elif op in ('AvgPool1dFwdOp','AvgPool2dFwdOp','AvgPool3dFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveAvgPool2dFwdOp','AdaptiveMaxPool2dFwdOp'):
  status='supported_untested';reason='Window layout selects original ReduceMax/ReduceMean; custom divisors select the original ReduceMeanWithCount API.';source=[evidence(SOURCE/('math/reduce_max/op_host/reduce_max_def.cpp' if 'Max' in op else 'math/reduce_mean/op_host/reduce_mean_def.cpp'))]
 elif op in ('INT8DequantPerTensorFwdOp','INT8DequantPerChannelFwdOp','INT8DequantPerBlockFwdOp'):
  status='supported_untested';reason='Original Mul with exact INT8-to-FP32 widening and original scale layout.';source=[evidence(SOURCE/'math/mul/op_host/mul_def.cpp')]
 elif op in ('GemmFwdOp','BmmFwdOp','GroupedGemmFwdOp','Conv1dFwdOp','Conv2dFwdOp','Conv3dFwdOp','SSDChunkCouplingFwdOp','MoEGroupedGemmFwdOp','MHCPostFwdOp'):
  status='supported_untested';reason='Original weighted-reduction DAG computes ReduceSum(x*count/count_sum); dot-product layout chooses count=weights and unit divisor, with no native math/schedule edits.';source=[evidence(SOURCE/'math/reduce_mean_with_count/op_kernel/arch35/reduce_mean_with_count_dag.h',15)]
 elif op in ('VectorNormFwdOp','CountNonzeroFwdOp'):
  status='supported_untested';reason='Original Cdist(x,zero,p) directly computes p-norm or zero-order count after reduction-axis layout.';source=[evidence(SOURCE/'math/cdist/op_host/cdist_def.cpp')]
 elif op in ('MaxPool1dIndicesFwdOp','MaxPool2dIndicesFwdOp','MaxPool3dIndicesFwdOp','AdaptiveMaxPool2dIndicesFwdOp'):
  status='supported_untested';reason='Original ArgMaxWithValue computes both outputs over packed valid windows; deterministic coordinate lookup restores original flat indices.';source=[evidence(SOURCE/'math/arg_max_with_value/op_graph/arg_max_with_value_proto.h',41)]
 elif op=='TopKSelectFwdOp':
  status='supported_untested';reason='Original TopKV2 over per-row valid-window crops; coordinate lookup and sentinel padding preserve the complete selection contract.';source=[evidence(SOURCE/'math/top_k_v2/op_graph/top_k_v2_proto.h',31)]
 elif op in COMPOSITIONS:
  status='supported_untested';reason='Same-library original native API composition; Torch supplies only storage, exact widening, layout and final casts. Every used primitive requires own native launch attribution.'
  source=[evidence(Path(__file__).with_name('composition.py'))]
 elif op in specific:
  status,rel,reason=specific[op];source=[evidence(SOURCE/rel)] if rel and (SOURCE/rel).exists() else [cat]
 else:
  status='unsupported_math';reason=reasons[info['category']];source=[cat,evidence(SOURCE/'docs/zh/op_list.md')]
 cases=[]
 for c in inv['cases']:
  if c['op']!=op:continue
  row={'case_id':c['case_id'],'status':status,'reason':reason,'receipt':None,'device_executed':False}
  limit=semantic_reason(op,c['params'])
  if limit and status=='supported_untested':row.update(status='unsupported_math',reason=limit,semantic_screen=evidence(Path(__file__).with_name('semantic_limits.py')))
  if blocker_active and row['status']=='supported_untested':
   row.update(status='deployment_blocked',reason=blocker['reason'],validation_blocker=str(blocker_file.relative_to(ROOT)),source_route_found=True,numerical_validation='not_run')
  if (op,c['case_id']) in historical:
   h=historical[(op,c['case_id'])];row.update(status='verified',reason='Historical complete manifest, edge checks and original root native-ownership acceptance; every source hash revalidated unchanged.',receipt=h['receipt'],acceptance=str(hf.relative_to(ROOT)),root_review=h['root_review'],reused_historical_device_evidence=True,device_executed=True)
  history=receipts.get((op,c['case_id']),[])
  if history:
   f,r=history[-1];row.update(receipt=str(f.relative_to(ROOT)),status='numerical_pass_unattributed' if r['status']=='verified' else 'unsupported_math' if r['status']=='unsupported' else 'numerical_failure' if r.get('error_type')=='AssertionError' and r.get('stage') in ('verify','edge_check') else 'deployment_blocked',reason=r.get('error',r.get('reason',r['status'])),device_executed=r.get('stage') in ('verify','edge_check'))
   row['attempts']=[str(p.relative_to(ROOT)) for p,_ in history]
  if (op,c['case_id']) in acceptances:
   f,r=acceptances[(op,c['case_id'])]
   if r.get('accepted') and history and r.get('receipt')==str(history[-1][0].relative_to(ROOT)):
    row.update(status='verified',reason='Complete original manifest output and input preservation passed; own-vendor native compilation/launch attributed.',acceptance=str(f.relative_to(ROOT)),device_executed=True)
  if row['status']=='verified' and history:
   recorded=history[-1][1]
   changed=[]
   for name,expected in (recorded.get('source_sha256',{})|recorded.get('oracle_source_sha256',{})).items():
    path=ROOT/name
    # Historical Abs was relocated verbatim; retain that explicit exception.
    if op=='AbsFwdOp' and 'isolated foreground process' not in recorded.get('api','') and name in ('TileOPs/benchmarks/ascend_all_pairs/ops_math/graph.py','TileOPs/benchmarks/ascend_all_pairs/ops_math/specs.py'):
     path=Path(__file__).parent/'validated_abs'/Path(name).name
    if not path.is_file() or sha(path)!=expected:changed.append(name)
   if changed:row.update(status='needs_revalidation',reason='Accepted historical receipt retained; current adapter/native deployment/oracle source differs.',changed_sources=changed)
  cases.append(row)
 if op=='AbsFwdOp':
  for row in cases:row['active_adapter_version']='.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_math/abs_active_version.json'
 counts=dict(Counter(c['status'] for c in cases))
 rows.append({**info,'status':next(iter(counts)) if len(counts)==1 else 'mixed' if counts else status,'reason':reason,'workload_factory':'missing' if op in inv.get('missing_factory_ops',[]) else 'inventoried','source':source,'adapter':'TileOPs/benchmarks/ascend_all_pairs/ops_math/validated_abs/graph.py' if op=='AbsFwdOp' else 'TileOPs/benchmarks/ascend_all_pairs/ops_math/graph.py' if op in SPECS else 'TileOPs/benchmarks/ascend_math_baselines.py' if op=='AddFwdOp' else 'TileOPs/benchmarks/ascend_all_pairs/ops_math/layernorm.py' if op in ('LayerNormFwdOp','AdaLayerNormFwdOp','GroupNormFwdOp','InstanceNormFwdOp','BatchNormFwdOp') else 'TileOPs/benchmarks/ascend_all_pairs/ops_math/mean_pooling.py' if op=='MeanPoolingFwdOp' else 'TileOPs/benchmarks/ascend_all_pairs/ops_math/topk_select.py' if op=='TopKSelectFwdOp' else 'TileOPs/benchmarks/ascend_all_pairs/ops_math/distance.py' if op in ('VectorNormFwdOp','CountNonzeroFwdOp') else 'TileOPs/benchmarks/ascend_all_pairs/ops_math/pool_indices.py' if 'Pool' in op and 'Indices' in op else 'TileOPs/benchmarks/ascend_all_pairs/ops_math/pooling.py' if 'Pool' in op and status=='supported_untested' else 'TileOPs/benchmarks/ascend_all_pairs/ops_math/dequant.py' if op in ('INT8DequantPerTensorFwdOp','INT8DequantPerChannelFwdOp','INT8DequantPerBlockFwdOp') else 'TileOPs/benchmarks/ascend_all_pairs/ops_math/weighted.py' if op in ('GemmFwdOp','BmmFwdOp','GroupedGemmFwdOp','Conv1dFwdOp','Conv2dFwdOp','Conv3dFwdOp','SSDChunkCouplingFwdOp','MoEGroupedGemmFwdOp','MHCPostFwdOp') else None,'api':SPECS.get(op,{}).get('graph'),'case_counts':counts,'cases':cases})
for row in rows:
 if row['op'] in COMPOSITIONS:row['adapter']='TileOPs/benchmarks/ascend_all_pairs/ops_math/composition.py'
 if row.get('adapter') and row['op']!='AddFwdOp':
  row['native_adapter']=row['adapter']
  row['adapter']='TileOPs/benchmarks/ascend_all_pairs/ops_math/isolated.py'
  row['deployment']='Clean checker environment; original native APIs in foreground same-guard process; complete host transfers included in timing.'
result={'schema':'ops-math-all-pairs/v1','updated_at':datetime.now(timezone.utc).isoformat(),'library':'ops-math','inventory_sha256':sha(EVIDENCE.parent/'inventory.json'),'manifest_sha256':inv['manifest_sha256'],'case_evidence_inheritance':'Each case inherits operator source file/line/hash and adapter; receipts additionally contain executed source hashes.','scope':{'operators':205,'cases':len(inv['cases'])},'status_definitions':{'verified':'complete manifest pass, input protection and own native runtime ownership accepted','supported_untested':'equivalent source API found, no accepted receipt','numerical_pass_unattributed':'numerical pass still lacking ownership acceptance','numerical_failure':'original manifest check failed','deployment_blocked':'build/API/runtime/device execution blockage','unsupported_math':'no currently audited permitted route for the full mathematical/output contract; this does not assert impossibility of future native compositions','needs_revalidation':'historical accepted evidence retained, but current adapter, deployment or oracle source changed','dependency_only':'public API delegates device calculation outside this library'},'operator_status_counts':dict(Counter(r['status'] for r in rows)),'case_status_counts':dict(Counter(c['status'] for r in rows for c in r['cases'])),'missing_workload_factories':[r['op'] for r in rows if r['workload_factory']=='missing'],'operators':rows}
(EVIDENCE/'coverage.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('operator_status_counts','case_status_counts','missing_workload_factories')}))
