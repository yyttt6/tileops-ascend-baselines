#!/usr/bin/env python3
"""CPU-only coverage report; never treats a static audit as a device pass."""
import ast, collections, hashlib, json, pathlib, sys
from datetime import datetime, timezone
sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[3]
WS = ROOT.parent
sys.path.insert(0, str(ROOT))
from benchmarks.ascend_all_pairs.tilekernels import PROVIDERS, DIAGNOSTIC_PROVIDERS
EVIDENCE = WS / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilekernels'
VENDOR = WS / 'ascend_baselines/TileKernels/tile_kernels'

def rel(p): return str(p.relative_to(WS))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')

source_index = []
for path in sorted(VENDOR.rglob('*.py')):
    parsed = ast.parse(path.read_text())
    source_index.append({'file': rel(path), 'line': 1, 'sha256': sha(path),
                         'reference_only': any(s in path.parts for s in ('torch', 'testing')),
                         'symbols': [{'name': n.name, 'line': n.lineno, 'end_line': n.end_lineno}
                                     for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]})
write(EVIDENCE/'source_inventory.json', {'schema': 'tilekernels-source-inventory/v1', 'files': source_index})
index = {s['file']: s for s in source_index}
def sources(*names):
    return [{k: v for k,v in index[rel(VENDOR/n)].items() if k != 'reference_only'} for n in names]

family_evidence = {
 'elementwise': ('quant/swiglu_forward_kernel.py', 'mhc/norm_fn_kernel.py', 'mhc/pre_split_mixes_kernel.py', 'moe/topk_gate_kernel.py', 'quant/norm_forward_kernel.py'),
 'normalization': ('quant/norm_forward_kernel.py', 'quant/norm_forward_asc.py'),
 'conv_pool': ('moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py', 'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py'),
 'reduction': ('moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py', 'mhc/norm_fn_asc.py'),
 'rope': ('transform/rope_kernel.py', 'transform/rope_asc.py'),
 'quantization': ('quant/common.py', 'quant/per_token_cast_asc.py', 'quant/per_block_cast_asc.py', 'quant/cast_back_kernel.py'),
 'moe': ('moe/moe_topk_gate_forward_kernel.py', 'moe/normalize_weight_kernel.py', 'quant/swiglu_forward_kernel.py'),
 'sampling': ('moe/topk_gate_kernel.py', 'moe/moe_topk_gate_forward_kernel.py', 'rand/randn_kernel.py'),
 'other': ('engram/engram_gate_fwd_kernel.py', 'mhc/pre_big_fuse_kernel.py', 'mhc/post_kernel.py'),
}
reasons = {
 'elementwise': 'No native entry implements this scalar operation. Original exported arithmetic is RMSNorm/residual addition, SwiGLU, affine sigmoid mixes, weight products and top-k selection; layout/padding/dtype/API selection cannot supply the missing function.',
 'normalization': 'Original normalization is RMSNorm (with optional residual/weight/FP8). No equivalent mean-centering, affine bias, batch statistics or required backward normalization contract is implemented.',
 'reduction': 'No equivalent original exported reduction result. Available raw reduction is sum of split partials, RMS/square-sum auxiliary and top-k indices; the requested reduction would require additional mathematical computation.',
 'conv_pool': 'No owned convolution API/kernel exists in the inspected native package. Engram gating omits its convolution; the available original reductions support the separately registered pooling layouts but do not provide a native weighted convolution contract.',
 'gemm': 'No TileKernels-owned matrix-product kernel exists in this checkout. mHC accepts already-computed GEMM partials; external DeepGEMM/Torch products cannot be credited to TileKernels.',
 'moe': 'Native routing supplies scores, selected expert IDs and normalized weights, but no owned expert GEMM/MLP or token permutation/scatter implementation supplies this full contract. External matrix products or Torch token routing cannot be credited to TileKernels.',
 'attention': 'No native attention/KV-cache/indexing implementation matches this full contract; rotation and top-k expert routing are partial components only.',
 'linear_attention': 'No DeltaNet/GLA/GDN/KDA state recurrence or chunked linear-attention kernel exists in the inspected original package.',
 'ssm': 'No causal convolution, selective scan or SSD/Mamba state-space recurrence is implemented by this native package.',
 'sampling': 'Native topk_gate returns at most 32 indices and rejects nonfinite input; no threshold/tie-preserving variable-k mask, top-p filter or categorical/speculative sampler exists. randn is Gaussian generation, not categorical sampling.',
 'quantization': 'Original native casts quantize/dequantize floating FP8/FP4 formats. The requested signed INT8 or asymmetric INT4 code/scaling contract is different; lossless layout/dtype mapping cannot change its arithmetic.',
 'other': 'No equivalent complete native operation exists. The package contains mHC mixing and Engram gates but no FFT/Hadamard or full Engram convolution/decode pipeline.',
}
special = {
 'FP8QuantPerBlockFwdOp': 'Manifest requires one scale per 128x128=16384 values and scale=1 for all-zero tiles. Native Ascend per_block_cast asserts 32x32=1024 values; per_token/per_channel groups have at most128/32 elements and clamp amax to1e-4. Padding cannot shrink a complete group; changing reductions/floor is forbidden.',
 'FusedTopKFwdOp': 'Original native sigmoid mix, top-k, residual sum and normalize_weight APIs supply sigmoid routing; no original standalone softmax score output exists for the softmax cases.',
 'TopKSelectFwdOp': 'Native topk_gate has hard num_topk<=32; manifest requests k=1024/2048 with per-row start/end masks. Changing native k limit/masking or kernel schedule is outside allowed adapters.',
 'MHCPreFwdOp': 'Native pre_big_fuse requires externally computed GEMM dot/square-sum partials, supports mhc4 only, and uses different RMS/Sinkhorn stabilization. Manifest needs phi projection and its complete three-output computation; package has no owned GEMM to supply it.',
 'EngramGateConvFwdOp': 'Native engram_gate_fwd performs signed-sqrt/clamped gating without causal convolution. Manifest uses sigmoid of a plain normalized dot plus depthwise convolution and SiLU.',
 'EngramGateConvBwdOp': 'Native engram_gate_bwd differentiates the signed-sqrt/clamped gate and does not differentiate the manifest causal convolution/SiLU pipeline.',
 'EngramDecodeFwdOp': 'Native Engram gate has no W_K/W_V projection, convolution-state update or decode convolution contract; a gate-only result is not the manifest output.',
 'SoftmaxFwdOp': 'mHC Sinkhorn starts with row softmax but always also performs column normalization, even repeat=0; no standalone full-row softmax result is exported.',
 'FusedAddRMSNormPerTokenQuantFwdOp': 'Manifest requires one scale per full row, optional scale upper bound, zero scale for all-zero rows, FP32 unrounded residual statistics and quantized INT8/FP8 outputs. Original norm quant API uses32-channel FP8 groups with positive amax floor and input-dtype residual statistics; no API selection supplies the complete contract.',
}

inv = read(EVIDENCE/'inventory_snapshot.json')
classic = [r for r in read(ROOT/'benchmarks/ascend/classic_matrix.json')['rows'] if r['library']=='TileKernels']
history = {(r['op'],r['case_id']):r for r in classic}
new = collections.defaultdict(list)
for p in sorted((EVIDENCE/'receipts').glob('*.json')):
    row=read(p)
    if 'op' in row and 'case_id' in row: new[(row['op'],row['case_id'])].append((p,row))
missing=set(inv.get('missing_factory_ops',[]))
# Missing factory status can change while the root agent supplies shared factories.
# Only use AST registrations; this report does not import Torch or initialize NPU.
for p in (ROOT/'benchmarks/_cases').glob('*.py'):
    tree=ast.parse(p.read_text())
    literals={n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value,str)}
    missing -= literals
rows=[]
ops=[]
for op in inv['operators']:
    name=op['op'];category=op['category']; factory=PROVIDERS.get(name) or DIAGNOSTIC_PROVIDERS.get(name)
    reason=special.get(name,reasons.get(category,reasons['other']))
    files=family_evidence.get(category, ('__init__.py',))
    ev=sources(*files)
    if factory:
        adapter=pathlib.Path(sys.modules[factory.__module__].__file__)
        line=next((n.lineno for n in ast.parse(adapter.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==factory.__name__),1)
        adapter_record={'file':rel(adapter),'line':line,'sha256':sha(adapter),'factory':factory.__name__}
    else: adapter_record=None
    op_rows=[]
    for c in inv['cases']:
        if c['op']!=name:continue
        key=(name,c['case_id']); old=history.get(key)
        row={'op':name,'case_id':c['case_id'],'status':'supported_unverified' if factory else 'unsupported_math',
             'reason':'Original native API adapter implemented; awaiting exact manifest/device verification.' if factory else reason,
             'workload_factory':'missing' if name in missing else 'available', 'adapter':adapter_record,
             'source_evidence':ev,'receipt':None,'device_validated':False,'test_history':[],
             'manifest_case':c}
        p=c.get('tensors',{})|c.get('params',{})
        if name in ('AllFwdOp','AnyFwdOp','CountNonzeroFwdOp') and p.get('x',[None,None])[1] not in ('bool','float16','bfloat16','float32'):
            row.update(status='unsupported_dtype',reason='Native nonzero selection requires an exact real FP32 representation; INT32/INT64/complex domains cannot be widened losslessly by this adapter.')
        if name == 'FusedTopKFwdOp' and p.get('scoring_func') != 'sigmoid':
            row.update(status='unsupported_math',reason='Native fused router only supports sqrtsoftplus; standalone native sigmoid exists, but no equivalent standalone softmax output is exposed.')
        if name == 'DivFwdOp' and p.get('rounding_mode') is not None:
            row.update(status='unsupported_math',reason='Native reciprocal/product has no floor/trunc output.')
        if name in ('ArgmaxFwdOp','ArgminFwdOp','AminFwdOp','AllFwdOp','AnyFwdOp') and row['status']=='supported_unverified':
            shape=p['x'][0]; dim=p.get('dim');axes=range(len(shape)) if dim is None else ([dim] if isinstance(dim,int) else dim)
            width=__import__('math').prod(shape[d] for d in axes)
            # Native six staged FP32 score rows exceed UB from width11008 upward.
            if width>=11008:
                row.update(status='deployment_blocked',reason=f'Native topk six-stage score UB alone is >=6*{width}*4 bytes, exceeding248KiB; same original kernel failure proven by historical Amax receipts. Full reduction cannot be split under the allowed schedule scope.',
                           blocking_evidence='.agent_memory/THREADS/tileops-progress-report-20261019/CLASSIC_BASELINE_PILOT/runs/classics_026/021-TileKernels.json',device_attempted=False)
        if old:
            row['historical_classic']=old
            receipt=old.get('device_receipt') or old.get('last_receipt')
            if old['status']=='verified' and receipt:
                evidence=read(WS/receipt)
                changed=[f for f,h in evidence.get('source_sha256',{}).items() if not (WS/f).exists() or sha(WS/f)!=h]
                if not changed:
                    row.update(status='verified',reason='Preserved historical full manifest, input-protection and root native-attribution acceptance; every recorded source hash still matches.',receipt=receipt,device_validated=True,verification_origin='historical_source_revalidated')
                else: row['historical_changed_sources']=changed
            elif factory and old['status']=='source_adapter_blocked_kernel_limit':
                row.update(status='deployment_blocked',reason=old['reason'],receipt=receipt,verification_origin='historical_failure_reaudited')
        if key in new:
            candidates=sorted(new[key], key=lambda a:a[1].get('finished_at',''))
            for rp,r in candidates:
                row['test_history'].append({'receipt':rel(rp),'status':r.get('status'),'stage':r.get('stage'),'error':r.get('error')})
            rp,r=candidates[-1]
            status=r.get('status')
            if status=='verified':
                calls=r.get('vendor_python_calls',[])
                own=any(v['file'].startswith('ascend_baselines/TileKernels/') and v['file'].endswith('_asc.py') for v in calls) or any(v.get('vendor_stack') and 'ascend' in v.get('target','').lower() for v in r.get('native_kernel_launches',[]))
                allout=r.get('verification',{}).get('checked_outputs')==r.get('verification',{}).get('total_outputs')
                changed=[f for f,h in r.get('source_sha256',{}).items() if not (WS/f).exists() or sha(WS/f)!=h]
                if own and allout and r.get('input_preservation_requested') and not changed:
                    row.update(status='verified',reason='Complete manifest outputs and input protection passed through the unchanged checker; traced original TileKernels Ascend entry.',device_validated=True,verification_origin='new_device_receipt')
                else: row.update(status='attribution_review_required',reason='Numerical receipt lacks complete native-call/input-protection acceptance or current source hashes differ.', changed_candidate_sources=changed)
            elif status=='unsupported':row.update(status='unsupported_dtype',reason=r.get('reason'))
            else:
                numerical=r.get('stage') in ('verify','edge_check') and r.get('error_type')=='AssertionError' and 'eps must be at least' not in str(r.get('error'))
                row.update(status='numerical_failed' if numerical else 'deployment_blocked',reason=r.get('error') or r.get('sync_error') or status)
            row.update(receipt=rel(rp),api=r.get('api'),adaptation=r.get('adaptation'),source_sha256=r.get('source_sha256'),verification=r.get('verification'))
            for field in ('reference_device','reference_engine','reference_placement_adaptation','reference_source_sha256','oracle_failure_diagnostic','oracle_failure_diagnostic_sha256'):
                if field in r: row[field]=r[field]
            if 'MaxPool' in name and (EVIDENCE/'original_pooling_oracle_diagnostic.json').exists():
                row['oracle_execution_issue']=rel(EVIDENCE/'original_pooling_oracle_diagnostic.json')
                for attempt in row['test_history']:
                    if attempt['status']=='device_sync_failed':
                        attempt['attribution_note']='Original NPU reference independently failed before any vendor call; this asynchronous error receipt alone cannot establish a vendor kernel fault. See original_pooling_oracle_diagnostic.json.'
        if name in missing and row['status']=='supported_unverified':
            row.update(status='missing_workload_factory',reason='Native mapping identified but no shared exact manifest workload factory is available.')
        rows.append(row);op_rows.append(row)
    counts=dict(collections.Counter(r['status'] for r in op_rows))
    ops.append({**op,'case_count':len(op_rows),'status':'mixed' if len(counts)>1 else next(iter(counts),'missing_workload_factory'),
                'status_counts':counts,'registered':name in PROVIDERS,'adapter':adapter_record,'source_evidence':ev,
                'reason':('Original native mapping implemented; see per-case receipts/constraints.' if factory else reason),
                'workload_factory':'missing' if name in missing else 'available'})
summary={'operators':len(ops),'cases':len(rows),'registered_operators':len(PROVIDERS),'diagnostic_operators':sorted(DIAGNOSTIC_PROVIDERS),
         'case_status_counts':dict(collections.Counter(r['status'] for r in rows)),
         'new_verified_cases':sum(r.get('verification_origin')=='new_device_receipt' and r['status']=='verified' for r in rows),
         'historical_verified_cases':sum(r.get('verification_origin')=='historical_source_revalidated' and r['status']=='verified' for r in rows),
         'original_reference_on_cpu_cases':sum(r.get('reference_device')=='cpu' and r['status']=='verified' for r in rows),
         'missing_factory_ops':sorted(missing)}
coverage={'schema':'tilekernels-all-pairs/v1','library':'TileKernels','updated_at':datetime.now(timezone.utc).isoformat(),
 'schema_notes':{'operators':'One row per manifest op; status mixed means per-case outcomes differ. Diagnostic-only factories are excluded from formal registration.',
 'cases':'One row per exact inventoried manifest case. Static conclusions are never device_validated.',
 'verified':'Unchanged full manifest checks + preserve_inputs + original vendor Ascend call tracing, or historical root acceptance with matching source fingerprints.',
 'deployment_blocked':'Native resource/compiler/runtime limit; distinct from mathematical absence. device_attempted false means inferred from current source plus cited same-kernel failure.',
 'numerical_failed':'Original unchanged manifest comparison failed.',
 'supported_unverified':'Implemented candidate without accepted real-device result.',
 'unsupported_math':'No equivalent original API under permitted layout/dtype/padding/API composition.',
 'unsupported_dtype':'No exact input representation in available native API.',
 'missing_workload_factory':'No exact shared Case construction, so no invented test.'},
 'provider_module':'benchmarks.ascend_all_pairs.tilekernels',
 'native_artifact_audit':rel(EVIDENCE/'native_artifact_audit.json'),
 'source_inventory':rel(EVIDENCE/'source_inventory.json'),'inventory_snapshot':rel(EVIDENCE/'inventory_snapshot.json'),
 'oracle_execution_issue':rel(EVIDENCE/'original_pooling_oracle_diagnostic.json') if (EVIDENCE/'original_pooling_oracle_diagnostic.json').exists() else None,
 'inventory_sha256':sha(EVIDENCE/'inventory_snapshot.json'),'summary':summary,'operators':ops,'cases':rows}
write(EVIDENCE/'coverage.json',coverage)
write(EVIDENCE/'progress.json',{'library':'TileKernels','worker':'6818450b8d604d58','updated_at':coverage['updated_at'],'stage':'device_validation','summary':summary,'coverage':rel(EVIDENCE/'coverage.json'),'next':'Run remaining supported cases under host npu-guard; inspect failures before retry.'})
print(json.dumps(summary,ensure_ascii=False))
