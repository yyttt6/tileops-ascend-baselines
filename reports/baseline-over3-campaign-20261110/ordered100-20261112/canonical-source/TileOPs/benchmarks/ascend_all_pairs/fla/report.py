"""CPU-only coverage regeneration from source audit, inventory and immutable receipts."""
import ast
import fcntl
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
EVIDENCE=ROOT/".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla"
VENDOR=ROOT/"ascend_baselines/flash-linear-attention-npu"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path,line=1):
    return dict(file=str(path.relative_to(ROOT)),line=line,sha256=digest(path))


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def main():
    inventory_path=EVIDENCE.parent/"inventory.json"
    inventory=json.loads(inventory_path.read_text())
    # Preserve the exact input used by this task, since root may refresh metadata.
    snapshot=EVIDENCE/("inventory-"+digest(inventory_path)[:12]+".json")
    if not snapshot.exists():snapshot.write_bytes(inventory_path.read_bytes())
    scope={}
    namespace={'__file__':str(HERE/'__init__.py')}
    exec(compile((HERE/"__init__.py").read_text(),str(HERE/"__init__.py"),"exec"),namespace)
    providers=namespace["PROVIDERS"]
    entry_tree=ast.parse((HERE/'__init__.py').read_text())
    entries={node.name:node for node in entry_tree.body if isinstance(node,ast.FunctionDef)}
    def adapter_evidence(op):
        factory=providers[op].__name__
        entry=entries[factory]
        evidence=[source(HERE/'__init__.py',entry.lineno)]
        visited={'__init__'}
        def add_module(module,names=()):
            if not module or module in visited:return
            path=HERE/(module+'.py')
            if not path.exists():return
            visited.add(module)
            tree=ast.parse(path.read_text())
            line=next((n.lineno for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names),1)
            evidence.append(source(path,line))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.level==1:
                    add_module(node.module,tuple(n.name for n in node.names))
        for node in ast.walk(entry):
            if isinstance(node,ast.ImportFrom) and node.level==1:
                add_module(node.module,tuple(n.name for n in node.names))
        return evidence
    adapters={op:adapter_evidence(op) for op in providers}
    public=VENDOR/"torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py"
    ctypes=public.with_name("_aclnn_ctypes.py")
    triton=VENDOR/"fla/ops/triton/triton_core"
    catalog=[]
    for path in [public,ctypes,*sorted(triton.glob("*.py"))]:
        tree=ast.parse(path.read_text())
        catalog.extend(dict(**source(path,node.lineno),api=node.name) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)))
    write_json(EVIDENCE/'source_catalog.json',catalog)
    api_source={r['api']:r for r in catalog}
    generic=[source(public,32),source(ctypes,1)]
    receipts={}
    for path in sorted((EVIDENCE/"runs").glob("*.json")):
        try:row=json.loads(path.read_text())
        except json.JSONDecodeError:
            # The unchanged checker may still be writing this fresh receipt;
            # never infer a result from incomplete bytes. A later refresh reads it.
            continue
        if row.get('kind')=='oracle_diagnostic':continue
        if 'op' in row and 'case_id' in row:
            receipts.setdefault((row['op'],row['case_id']),[]).append((path,row))
    histories=json.loads((ROOT/"TileOPs/benchmarks/ascend/classic_matrix.json").read_text())['rows']
    histories={(r['op'],r['case_id']):r for r in histories if r['library']=='flash-linear-attention-npu'}
    reasons={
        'elementwise':'No equivalent exported original FLA pointwise operation. Own kernels provide convolution/SiLU, L2 norm, scans and linear-attention recurrences; their embedded arithmetic does not expose this operation via layout or API parameters.',
        'reduction':'No equivalent original FLA reduction contract: cumsum exposes additive prefixes; L2Norm produces normalized values/rstd, not this reduction. No max/min/order/logical/product/softmax reducer is exported.',
        'normalization':'No original FLA entry for this centering/statistics/affine normalization contract. L2Norm has uncentered sum-of-squares normalization only.',
        'conv_pool':'No original FLA maximum/index-selection pool. Causal convolution and cumsum have sum semantics; layout cannot make them return extrema or arg-indices.',
        'gemm':'No complete generic GEMM entry in original FLA. ChunkFwdO is a causal-attention+state suboperator with fixed feature/layout contracts; it is not a generic GEMM API. No verified complete shape-only lowering is available.',
        'quantization':'FLA original exports have no quantizer/dequantizer scale, rounding, packing or output-scale contract.',
        'attention':'FLA delta/KDA are linear recurrent attention. They do not provide manifest softmax attention, paging, sparse selection or KV-cache operation.',
        'moe':'FLA original exports do not implement token/expert routing, permutation, gating or complete expert MLP contracts.',
        'sampling':'FLA original exports have no categorical sampling, top-k/top-p or probability-mask implementation.',
        'linear_attention':'Original ChunkFwdH(gk,W=0) can express the GLA forward state recurrence; no complete original API pipeline for the requested gradients/saved outputs has been matched.',
        'ssm':'FLA delta/KDA updates differ from SSD/selective-scan recurrence and chunk intermediate contracts; no complete equivalent own SSM API exported.',
        'rope':'No owned FLA rotary-position kernel or complete coefficient-generation/rotation API in the inspected source tree; KDA gates and recurrent states do not implement rotary phase pairs.',
        'other':'No original FLA API with the requested transform/cache/embedding contract.'}
    special={
        'SSDChunkCumsumFwdOp':('deployment_blocked','Original KdaGateCumsum softplus plus original native tap product and chunk_local_cumsum compose both declared outputs; sequence.py implemented, conditional registration awaits the owned gate deployment.'),
        'SoftplusFwdOp':('deployment_blocked','Original owned KdaGateCumsum raw gate at T=1 and A_log=0 supplies softplus with a constant base-two scale conversion; gate_activations.py implemented, but the original native package is still building and no executable provider is registered.'),
        'GeluFwdOp':('unsupported_mathematics','FastGelu public wrapper uses aclnnFastGelu, but no owned FastGelu kernel exists in this vendor tree/deployments; using stock CANN dependency is prohibited and its sigmoid approximation does not equal erf GELU.'),
        'GeluAndMulFwdOp':('unsupported_mathematics','No owned erf-GELU implementation; stock dependency FastGelu cannot substitute.'),
        'GeluTanhAndMulFwdOp':('unsupported_mathematics','No owned tanh-GELU implementation; FastGelu sigmoid approximation is different.'),
        'DeltaNetChunkFwdOp':('deployment_blocked','Current shared oracle now checks all six outputs and strict-lower WY signal. Complete adapter uses original prepare plus native FP32 tap/scan recurrence; conditional registration awaits the owned prepare deployment. Earlier Partial(1) finding is superseded by current source, with historical hashes retained.'),
        'GLAChunkFwdOp':('supported_unverified','Current shared oracle now checks both outputs with nonzero signal alignment. Original BF16 GLA recurrence adapter can serve this contract; exact-case execution remains required.'),
        'DeltaNetChunkBwdOp':('deployment_blocked','Original GDN backward can specialize g=0, but selected audited deployments do not define aclnnChunkGatedDeltaRuleBwd; original forward saved-buffer API and full gradients require further deployment work.'),
        'CausalConv1dFwdOp':('missing_workload_factory','Original causal API exists; all-pairs inventory supplies zero cases/workload factory. No test fabricated.'),
        'CausalConv1dUpdateFwdOp':('missing_workload_factory','Original stateful causal update API exists; inventory supplies zero cases/workload factory. No test fabricated.'),
    }
    rows=[]
    operators=[]
    for operator in inventory['operators']:
        op=operator['op']
        own_cases=[r for r in inventory['cases'] if r['op']==op]
        registered=op in providers
        if registered:
            state,reason='supported_unverified','Original API adapter implemented; numerical execution is required for each exact manifest case.'
            if op in ('ClampScalarFwdOp','ClampTensorFwdOp','MaximumFwdOp','MinimumFwdOp'):
                reason='Finite-real magnitude identity implemented using original APIs, but the unchanged elementwise oracle requires atol=rtol=0. Original reciprocal-L2 approximation and FP32 cancellation have not been certified against that exact contract; do not infer a pass from algebra.'
        else:
            state,reason=special.get(op,('no_equivalent_original_api',reasons.get(operator['category'],reasons['other'])))
        if not own_cases:
            state='missing_workload_factory'
            reason=special.get(op,(state,'No workload case/factory present in shared inventory; source semantic conclusion does not fabricate a device test.'))[1]
        manifest=ROOT/'TileOPs'/operator['manifest']
        manifest_text=manifest.read_text().splitlines()
        section_start=next(i for i,line in enumerate(manifest_text) if line==op+':')
        section_end=next((i for i in range(section_start+1,len(manifest_text)) if manifest_text[i] and not manifest_text[i].startswith((' ','#'))),len(manifest_text))
        contract_end=next((i for i in range(section_start+1,section_end) if manifest_text[i].startswith('  workloads:')),section_end)
        contract_lines=manifest_text[section_start:contract_end]
        ref_api=next((line.split(':',1)[1].strip().strip('"\'') for line in contract_lines if line.startswith('  ref_api:')),None)
        manifest_source=source(manifest,section_start+1)
        sources=generic
        if op in ('CumsumFwdOp','SumFwdOp','MeanFwdOp','MeanPoolingFwdOp','AdaptiveAvgPool2dFwdOp','AvgPool1dFwdOp','AvgPool2dFwdOp','AvgPool3dFwdOp'):
            sources=[api_source['chunk_local_cumsum'],api_source['chunk_local_cumsum_scalar_kernel']]
        elif op in ('ReluFwdOp','LeakyReluFwdOp','PreluFwdOp','HardtanhFwdOp','HardsigmoidFwdOp','HardswishFwdOp','FusedQKNormRoPEFwdOp','AbsFwdOp','StdFwdOp','VectorNormFwdOp','BatchNormFwdOp','BatchNormBwdOp','RMSNormFwdOp','FusedAddRMSNormFwdOp','LayerNormFwdOp','FusedAddLayerNormFwdOp','AdaLayerNormFwdOp','AdaLayerNormZeroFwdOp','GroupNormFwdOp','InstanceNormFwdOp'):
            sources=[api_source['l2norm_fwd'],api_source['causal_conv1d_fwd_impl_old']]
        elif op in ('INT8DequantPerTensorFwdOp','INT8DequantPerChannelFwdOp','INT8DequantPerBlockFwdOp','HadamardTransformFwdOp','GroupedGemmFwdOp','AddFwdOp','SubFwdOp','MulFwdOp','NegFwdOp','SiluFwdOp','SiluAndMulFwdOp','LerpScalarFwdOp','LerpTensorFwdOp','Conv1dFwdOp','Conv2dFwdOp','Conv3dFwdOp'):
            sources=[api_source['causal_conv1d_fwd_impl_old'],api_source['causal_conv1d_fwd_kernel_old']]
        elif op in ('SigmoidFwdOp','TanhFwdOp'):
            sources=[api_source['npu_recurrent_kda'],api_source['npu_kda_gate_cumsum'],api_source['causal_conv1d_fwd_impl_old']]
        elif op in ('SoftplusFwdOp','SSDChunkCumsumFwdOp'):
            sources=[api_source['npu_kda_gate_cumsum'],api_source['causal_conv1d_fwd_impl_old']]
            if op=='SSDChunkCumsumFwdOp':sources.append(api_source['chunk_local_cumsum'])
        elif op in ('DeltaNetChunkFwdOp','DeltaNetChunkBwdOp'):
            sources=[api_source['npu_chunk_gated_delta_rule_fwd_prepare'],api_source['npu_chunk_gated_delta_rule_bwd']]
            if op=='DeltaNetChunkFwdOp':sources=[api_source['npu_chunk_gated_delta_rule_fwd_prepare'],api_source['causal_conv1d_fwd_impl_old'],api_source['chunk_local_cumsum']]
        elif op in ('ExpFwdOp','Expm1FwdOp'):
            sources=[api_source['npu_chunk_fwd_h'],api_source['causal_conv1d_fwd_impl_old']]
        elif op in ('GLAFwdOp','GLARecurrentFwdOp','GLAChunkFwdOp'):
            sources=[api_source['npu_chunk_fwd_h'],api_source['causal_conv1d_fwd_impl_old']]
        elif op in ('VarFwdOp','VarMeanFwdOp'):
            sources=[api_source['chunk_local_cumsum'],api_source['causal_conv1d_fwd_impl_old']]
        elif op in ('GemmINT8W8A8FwdOp','BmmFP8FwdOp','GemmFP8FwdOp','BmmFwdOp','GemmFwdOp'):
            sources=[api_source['npu_chunk_fwd_h'],api_source['npu_chunk_fwd_o'],api_source['causal_conv1d_fwd_impl_old']]
        elif op in ('CausalConv1dFwdOp','CausalConv1dUpdateFwdOp'):
            sources=[api_source['npu_causal_conv1d']]
        elif op in ('GDNFwdOp','DeltaNetFwdOp','KDAFwdOp','DeltaNetRecurrentFwdOp'):
            sources=[api_source['npu_chunk_gated_delta_rule_fwd'],api_source['npu_recurrent_kda']]
        if op in ('Conv1dFwdOp','Conv2dFwdOp','Conv3dFwdOp','GroupedGemmFwdOp'):
            sources=sources+[api_source['npu_chunk_fwd_h']]
        if op in ('MHCPostFwdOp','SSDStatePassingFwdOp','SSDChunkCouplingFwdOp'):
            sources=[api_source['causal_conv1d_fwd_impl_old']]
            if op!='MHCPostFwdOp':sources.append(api_source['npu_chunk_fwd_h'])
        if op in ('WhereFwdOp','MaskedFillScalarFwdOp','MaskedFillTensorFwdOp'):
            sources=[api_source['causal_conv1d_fwd_impl_old']]
        if op in ('ClampScalarFwdOp','ClampTensorFwdOp','MaximumFwdOp','MinimumFwdOp'):
            sources=[api_source['l2norm_fwd'],api_source['causal_conv1d_fwd_impl_old']]
        op_rows=[]
        for case in own_cases:
            row=dict(op=op,case_id=case['case_id'],status=state,reason=reason,source=sources,
                     adapter_path=str((HERE/'__init__.py').relative_to(ROOT)) if registered else None,
                     adapter_entry=adapters[op][0] if registered else None,
                     adapter_sources=adapters.get(op,[]),
                     registered=registered,device_validated=False,numerical_receipt=None,manifest_source=manifest_source)
            tensors=case.get('tensors',case['params'])
            if op in ('GLAFwdOp','GLARecurrentFwdOp','GLAChunkFwdOp') and tensors['q'][1]!='bfloat16':
                row.update(status='original_api_dtype_limit',reason='Original ChunkFwdH gk recurrence requires BF16; FP16 conversion would be lossy. This is a native dtype contract, not a different GLA formula.')
            if op in ('DeltaNetChunkFwdOp','DeltaNetChunkBwdOp') and tensors['q'][1]!='bfloat16':
                row.update(status='original_api_dtype_limit',reason='Original FLA prepare/backward requires BF16. FP16-to-BF16 conversion would be lossy; no conversion permitted.')
            if op=='VectorNormFwdOp' and case['params'].get('ord',2) not in (1,2):
                row.update(status='no_equivalent_original_api',reason='Original reciprocal-L2 composition supports ord=2 and scalar-L2 plus original sum supports ord=1; no original maximum reduction API matched for infinity.')
            if op=='KDAFwdOp' and tensors['q'][1]!='bfloat16':
                if (EVIDENCE/'kda_native_ready.json').is_file():
                    row.update(status='supported_unverified',reason='Exact original FP16/F32-gate KDA variant has been compiled and deployed with source/binary lineage; complete manifest numerical execution still required.')
                else:
                    row.update(status='deployment_blocked',reason='Original chunk KDA supports FP16 but is not present in the selected audited native package. Recurrent KDA is BF16-only; lossy casts forbidden.')
            if op=='GDNFwdOp' and case['params'].get('use_gate_in_kernel'):
                row.update(status='supported_unverified' if tensors['q'][1]=='bfloat16' else 'deployment_blocked',reason='Raw gate rejected by ChunkGatedDeltaRuleFwd; BF16 uses native recurrent KDA scalar-gate broadcast. FP16 needs an owned chunk-KDA deployment.')
            if op.startswith('Conv') and registered:
                w=tensors['weight'][0]
                width=1
                for s in w[1:]:width*=s
                if width>64:row.update(status='supported_unverified',reason=f'Window dot-product semantics match; {width} taps compose original <=64-tap calls with FP32 native residual accumulation. Exact-case runtime/resource and numerical checks pending.')
            history=histories.get((op,case['case_id']))
            if history:
                row['historical_status']=history['status']
                row['historical_receipt']=history.get('device_receipt',history.get('last_receipt'))
                if op=='Conv2dFwdOp' and history['status']=='verified':
                    row['historical_exclusion']='Historical adapter changed host BT/BD. Current task forbids changing native schedule; this new adapter must pass independently.'
            attempts=sorted(receipts.get((op,case['case_id']),[]),key=lambda item:item[1].get('started_at',item[1].get('timestamp',str(item[0].stat().st_mtime))))
            row['attempts']=[str(p.relative_to(ROOT)) for p,_ in attempts]
            if attempts:
                path,result=attempts[-1]
                status=result['status']
                if status=='verified':
                    check=result.get('verification',{})
                    events=result.get('call_attribution',[])
                    if result.get('attribution_scope')=='candidate_only':
                        owned=any(e.get('execution_scope')=='candidate' and ('flash-linear-attention-npu' in e.get('jit_source','') or (e.get('runtime_return_observed') and '/fla_npu/opp/' in e.get('actual_launch_owner',''))) for e in events)
                    else:
                        owned=any('flash-linear-attention-npu' in e.get('jit_source','') or e.get('native_symbol','').startswith('aclnn') for e in events)
                    full=check.get('checked_outputs')==check.get('total_outputs') and check.get('checked_outputs',0)>0
                    protected=result.get('input_preservation',{}).get('passed') or result.get('verifier_sha256')=='b26a3c9a1da702e5ceb0fadb124bcf3ef2985e4a930fcd6942527c42e6b57630'
                    status='verified' if owned and full and protected else 'attribution_pending'
                    row['input_protection_evidence']=result.get('input_preservation') or dict(verifier_sha256=result.get('verifier_sha256'),source='TileOPs/scripts/ascend_classic_check.py',line=292,mechanism='Request(preserve_inputs=True), original checker verified status after complete outputs and special checks')
                elif status=='unsupported':
                    detail=result.get('reason','')
                    status='original_api_dtype_limit' if any(x in detail for x in ('BF16','bfloat16','lossy','dtype')) else 'adaptation_incomplete'
                elif status in ('failed','device_sync_failed'):
                    status='numerical_failed' if result.get('error_type')=='AssertionError' else ('deployment_blocked' if result.get('stage')=='build' else 'execution_failed')
                    if 'feature dim >= 64KB' in result.get('error',''):status='original_api_shape_limit'
                    if 'JSON configuration file' in result.get('error','') or 'Wrong FLA package' in result.get('error',''):status='deployment_blocked'
                    if 'produced_by_reference = reference(*inputs)' in result.get('traceback',''):status='reference_execution_blocked'
                if result['status']=='unsupported' and op=='VectorNormFwdOp' and case['params'].get('ord',2) not in (1,2):status='no_equivalent_original_api'
                row.update(status=status,reason=result.get('error',result.get('reason','Original complete checker passed with input preservation and owned launch evidence.')),
                           numerical_receipt=str(path.relative_to(ROOT)),device_validated=status=='verified')
                row['verified_source_scope']='Immutable receipt and its adapter source snapshots; current source is listed separately.'
                row['tested_adapter_sha256']={k:v for k,v in (result.get('source_sha256',{})|result.get('isolated_provider_sha256',{})).items() if k.startswith('TileOPs/benchmarks/ascend_all_pairs/fla/')}
            rows.append(row);op_rows.append(row)
        counts=Counter(r['status'] for r in op_rows)
        operators.append(dict(op=op,category=operator['category'],status=(next(iter(counts)) if len(counts)==1 else 'mixed') if counts else state,
                              case_counts=dict(counts),case_count=len(op_rows),registered=registered,reason=reason,source=sources,manifest_source=manifest_source,
                              manifest_ref_api=ref_api,manifest_contract_excerpt='\n'.join(contract_lines)))
    out=dict(schema='fla-all-pairs-coverage/v1',library='flash-linear-attention-npu',
        schema_notes='operators: all 205 exact names; cases: exact shared-inventory workload rows; verified requires receipt full-output counts, unchanged input protection, and actual owned library dispatch. supported_unverified, reference_incomplete, reference_execution_blocked, missing_workload_factory, numerical_failed, deployment_blocked, execution_failed, adaptation_incomplete, original_api_dtype_limit, original_api_shape_limit, unsupported_mathematics, no_equivalent_original_api are separate. No-equivalent means no complete inspected own API within layout/API adaptation scope; unsupported_mathematics is reserved for a concrete differing formula. Neither is a proof of computational impossibility.',
        updated_at=datetime.now(timezone.utc).isoformat(),inventory_source=source(inventory_path),
        native_provenance={name:str((EVIDENCE/name).relative_to(ROOT)) for name in ('native_build_source_audit.json','native_binary_lineage.json','causal_embedded_source_audit.json','native_launch_audit.json','partial_native_ready.json','kda_native_ready.json') if (EVIDENCE/name).is_file()},
        oracle_issues=str((EVIDENCE/'oracle_issues.json').relative_to(ROOT)),
        provider_module='benchmarks.ascend_all_pairs.fla',source_catalog=str((EVIDENCE/'source_catalog.json').relative_to(ROOT)),
        counts=dict(Counter(r['status'] for r in rows)),operator_count=len(operators),case_count=len(rows),operators=operators,cases=rows)
    write_json(EVIDENCE/'coverage.json',out)
    write_json(EVIDENCE/'progress.json',{k:out[k] for k in ('updated_at','counts','operator_count','case_count','provider_module')}|dict(stage='implementation_and_device_validation',registered_operators=list(providers),verified_operators=sorted({r['op'] for r in rows if r['device_validated']}),numerical_receipt_count=sum(len(v) for v in receipts.values()),active_jobs=str((EVIDENCE/'active_jobs.json').relative_to(ROOT)),missing_workload=[r['op'] for r in operators if r['case_count']==0]))
    print(json.dumps({k:out[k] for k in ('operator_count','case_count','counts')}))


if __name__=='__main__':
    # Own report serialization only; this is unrelated to the host NPU lock.
    with (EVIDENCE/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
