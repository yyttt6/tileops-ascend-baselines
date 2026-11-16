"""CPU-only all-205 source conclusions and independently traceable case coverage."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True

ROOT=Path(__file__).resolve().parents[4]
EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl'
sys.path.insert(0,str(ROOT/'TileOPs'))
from benchmarks.ascend_all_pairs.sgl import PROVIDERS
from benchmarks.ascend_all_pairs.sgl.eligibility import restriction

PKG='ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/'
GROUPS={
 'elementwise':['activation/fused_sigmoid_mul.py','activation/swiglu_quant.py','activation/swiglu_oai.py','activation/situ.py','moe/mul_add.py','norm/scale_shift.py','fla/fused_gdn_gating.py'],
 'norm':['fla/layernorm_gated.py','norm/rmsnorm_bias.py','norm/add_rmsnorm_bias.py'],
 'reduction':['sample/argmax_softmax_prob.py','fla/layernorm_gated.py','fla/cumsum.py','norm/l1_norm.py'],
 'sampling':['sample/probability.py','sample/chain_speculative_sampling.py'],
 'rope':['norm/fused_rope_qk_mqa.py','norm/split_qkv_rmsnorm_rope.py'],
 'linear_attention':['fla/chunk.py','fla/chunk_kda_fwd.py','fla/kda_target_verify.py','fla/fused_sigmoid_gating_recurrent.py'],
 'mamba':['mamba/causal_conv1d.py','mamba/mamba_state_update_triton.py','mamba/speculative_state_scatter.py'],
 'attention':['indexer/flash_block_score_prefill.py','indexer/flash_block_score_decode.py','attention/decode_attention.py','attention/sinks_attention.py','attention/gqa_share_sparse_attention.py','attention/fia_blockq_attention.py'],
 'moe':['moe/zero_experts_compute_identity.py','activation/swiglu_quant.py'],
 'quantization':['activation/swiglu_quant.py','activation/swiglu_mxfp8_quant.py','activation/situ_mxfp8_quant.py','norm/rmsnorm_bias.py','norm/scale_shift.py'],
 'convolution':['mamba/causal_conv1d.py'],
 'pool':['sample/argmax_softmax_prob.py','fla/layernorm_gated.py'],
 'sequence_modeling':['kimi_k3/attn_residual.py','mamba/causal_conv1d.py'],
}
REASONS={
 'elementwise':'The inspected public SGL forms are scalar multiply-add, sigmoid-multiply, SwiGLU/SiTU, scale-shift and normalization. No equivalent public operator for {op} is exposed; extracting inner expressions or replacing target arithmetic in Torch would exceed this adapter scope.',
 'norm':'SGL exposes centered LayerNorm/RMSNorm and fused residual forms, but no matching running-statistics/gradient contract for {op}. Native return statistics and updates do not provide all declared outputs.',
 'reduction':'Native argmax exposes an index and peak softmax probability; LayerNorm exposes mean and reciprocal standard deviation; l1_norm returns normalized values. None exposes the complete {op} contract without additional arithmetic stages or changing native math.',
 'sampling':'SGL top_k/top_p APIs filter probability ranks and renormalize. {op} requires masking original logits, preserving retained bits and boundary ties; existing probability API cannot supply that contract by layout/dtype adaptation.',
 'linear_attention':'SGL kernels implement delta/KDA state updates. {op} has a different recurrence or backward-gradient outputs, not supplied by the existing forward pipeline; no unchanged own-library equivalent was identified.',
 'mamba':'The SGL mamba state module implements cache gather/scatter/rollback, and causal_conv1d handles per-channel temporal convolution. Neither supplies {op} SSD/selective-scan state mathematics and all outputs.',
 'attention':'SGL has decode MLA/GQA and sparse-selection attention APIs; no unchanged complete {op} contract was identified, including its compression/indexing/backward outputs. Calls delegated to torch_npu FIA are not attributed as SGL-owned arithmetic.',
 'moe':'The standalone SGL route does not expose the complete {op} routing/combining or FP32-intermediate MLP contract. Its own distributed DeepEP paths require multiple ranks/devices; dependency operators and Torch arithmetic are not substitutes.',
 'quantization':'SGL quantization entries first execute SwiGLU/SiTU or RMS normalization, and MXFP8 uses grouped exponent scales. The direct {op} scale/zero-point/rounding contract is not an optional identity path of those APIs.',
 'convolution':'SGL causal_conv1d is per-channel with temporal taps; {op} manifest requires spatial/cross-channel convolution. Dense LoRA only directly expresses pointwise kernels; general im2col/convolution construction is outside the accepted layout-only route.',
 'pool':'SGL has scalar argmax and mean statistics but no spatial/window or adaptive pooling API for {op}. Implementing window traversal and reduction as a new pooling algorithm exceeds the permitted adaptation route.',
 'gemm':'SGL own LoRA vector kernels accept FP16/BF16 operands and FP32 accumulation. {op} requires packed/FP8/INT quantization or block scales absent from that entry; optional CATLASS calls are dependency implementations, not SGL-owned kernels.',
 'fft':'No complex FFT transform in original public Python/native registrations; representing a DFT with GEMM would be a new algorithm, not a layout adaptation.',
 'transform':'No original SGL Hadamard transform implementation in the inspected public Python/native registration inventory.',
 'sequence_modeling':'SGL Kimi attention-residual uses RMS scores/softmax mixing. {op} requires Engram gating/convolution or MHC Sinkhorn matrices and outputs not supplied by that API.',
}

# A failed earlier entry remains evidence, but does not classify a subsequently
# implemented different public-API route as if that new route had been run.
PENDING_ALTERNATIVES={
 ('FusedAddRMSNormFwdOp','llama-405b-decode-bfloat16'):'norm_compat.py',
 ('FusedAddRMSNormFwdOp','llama-405b-prefill-bfloat16'):'norm_compat.py',
 ('InstanceNormFwdOp','image-track-stats-float16'):'norm_compat.py',
 ('MeanFwdOp','mlp-intermediate-float32'):'mean_compat.py',
 ('MeanFwdOp','long-seq-bfloat16'):'row_gemv.py',
}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    f=ROOT/path
    s=f.read_text(errors='replace')
    return {'file':path,'line_start':1,'line_end':len(s.splitlines()),'sha256':sha(f)}


def receipt_reason(row):
    reason=row.get('error') or row.get('reason') or 'Full original manifest outputs, controls/probes and preserved inputs passed; source ownership traced.'
    if len(reason)<4000:return reason
    diagnostics=[line for line in reason.splitlines() if 'error:' in line.lower() or 'overflow' in line.lower()]
    return '\n'.join(diagnostics[-6:])+'\nFull error and compiler IR remain in the unchanged numerical receipt.' if diagnostics else reason[:1500]+'\nFull error remains in the unchanged numerical receipt.'


def main():
    invpath=EVIDENCE/'inventory_snapshot_v2.json'
    inv=json.loads(invpath.read_text())
    classic=json.loads((ROOT/'TileOPs/benchmarks/ascend/classic_matrix.json').read_text())
    old={(r['op'],r['case_id']):r for r in classic['rows'] if r['library']=='sgl-kernel-npu'}
    receipts={}
    for f in sorted((EVIDENCE/'receipts').rglob('*.json')):
        try:r=json.loads(f.read_text())
        except (ValueError,OSError):continue
        key=(r.get('op'),r.get('case_id'))
        if key not in receipts or r.get('finished_at','')>receipts[key][1].get('finished_at',''):
            receipts[key]=(f,r)
    incomplete_attempts={}
    contaminated_batches=set()
    for result_path in (EVIDENCE/'campaigns').glob('*/results.json'):
        for attempt in json.loads(result_path.read_text()):
            rc=attempt['exit_code']
            if rc not in (75,76,78,124):continue
            log=ROOT/attempt['log']
            starts=[]
            for line in log.read_text(errors='replace').splitlines():
                try:entry=json.loads(line)
                except ValueError:continue
                if isinstance(entry,dict) and entry.get('phase')=='case_start':starts.append((entry['op'],entry['case_id']))
            if rc==76:contaminated_batches.add(attempt['batch'])
            if starts:
                incomplete_attempts[starts[-1]]=attempt
            elif rc in (75,78):
                for c in json.loads((ROOT/attempt['queue']).read_text()):
                    incomplete_attempts[c['op'],c['case_id']]=attempt
    files={}
    source_groups={}
    for family,paths in GROUPS.items():
        paths=[PKG+p for p in paths]
        source_groups[family]=paths
        for p in paths:
            if (ROOT/p).exists():files[p]=source(p)
    native='ascend_baselines/sgl-kernel-npu/csrc/pytorch_extensions.cpp'
    files[native]=source(native)
    operators=[];rows=[]
    missing=set(inv.get('missing_factory_ops',[]))
    reference_blocked=set(inv.get('reference_blocked_ops',[]))
    for spec in inv['operators']:
        op,family=spec['op'],spec['family']
        own=PROVIDERS.get(op)
        evidence=[files[p] for p in source_groups.get(family,[]) if p in files]+[files[native]]
        if own:
            af=Path(inspect.getsourcefile(own));adapter=str(af.relative_to(ROOT))
            fnline=inspect.getsourcelines(own)[1]
            reason='Original-library equivalent adapter implemented; every manifest case requires individual device/reference/ownership acceptance.'
            semantic='supported'
        else:
            adapter=None;fnline=None
            reason=REASONS.get(family,'No matching own-library complete public API for {op} in inspected source.').format(op=op)
            semantic='unsupported_within_scope'
        if op=='FP8LightningIndexerFwdOp' and not own:
            reason='Original lightning_indexer native API returns selected integer indices, while the manifest declares the complete FP32 [B,S,S_kv,G] weighted-ReLU logits (and per-key scales). Original block-score APIs use max/logsumexp summaries, not this complete weighted-ReLU score contract.'
        if op=='NSACompressedVarlenFwdOp':
            reason='Manifest requires both attention and natural-log LSE. Original decode_gqa/decode_mla expose only attention; own sparse_attn_sharedkv host explicitly rejects return_softmax_lse=true at csrc/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv.cpp:71. Dropping the LSE output is not acceptable.'
            source_path='ascend_baselines/sgl-kernel-npu/csrc/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv.cpp'
            evidence.append(source(source_path))
        if op in ('MLAPagedFwdOp','MLAVarlenFwdOp'):
            reason='The now-available original manifest oracle requires attention plus LSE (base2 for paged MLA, natural log for varlen MLA). SGL decode_mla exposes only attention; sparse_attn_sharedkv rejects return_softmax_lse=true. Original sgl_sparse_flash_attention returns softmax_max and softmax_sum, not LSE; no complete public logarithm stage was identified to form the required LSE without external arithmetic.'
            evidence.append(source('ascend_baselines/sgl-kernel-npu/csrc/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv.cpp'))
            evidence.append(source('ascend_baselines/sgl-kernel-npu/csrc/sparse_flash_attention/op_host/sparse_flash_attention.cpp'))
        if op=='FusedAddRMSNormPerTokenQuantFwdOp':
            reason='The completed manifest requires FP32 residual accumulation, prescribed storage roundings, optional scale upper bound and exact zero-scale rows. Original SGL fused quantization has a different scale/floor contract; layout or dtype changes cannot replace that arithmetic or omit any declared output.'
        if op=='PagedKVCacheGatherFwdOp':
            semantic='supported' if own else 'deployment_blocked'
            reason='Own unidex_copy has exact arbitrary src/dst row index semantics, but the current A5 binary lacks its A3-gated symbol. An isolated lazy-build factory compiles unchanged original host/kernel and registration-only binding into the worker cache. Optional lossless FP8 widening, original SGL scale multiplication and declared output conversion precede indexed copy. Acceptance requires real build and full device receipts.'
            evidence.append(source('ascend_baselines/sgl-kernel-npu/csrc/sparsity_driven_kv_offload/unidex_copy/op_host/unidex_copy.cpp'))
            evidence.append(source('ascend_baselines/sgl-kernel-npu/csrc/sparsity_driven_kv_offload/unidex_copy/op_kernel/unidex_copy_kernel.cpp'))
            if (EVIDENCE/'unidex_cpu_build_initial.json').exists():
                reason='Original unidex_copy host/kernel compiled unchanged for A5 with a registration-only bridge; offline build receipt records no NPU device access. Original SGL scaling supports optional FP8 inputs before native indexed byte copy. Shared-library loading and complete numerical acceptance are still required under npu-guard.'
        if op=='DSAPagedFwdOp':
            reason='Completed manifest uses the V3.2 656-byte cache row with four per-token FP32 FP8 scales plus BF16 RoPE tail, and requires natural-log LSE. No inspected original SGL API supplies this complete cache decode plus LSE; sparse flash exposes max/sum statistics rather than the required LSE.'
        if op=='NSATopKVarlenFwdOp':
            reason='Manifest ranks per-group sums of exp(score-LSE), with closed-block normalization and special priority for current/previous/first blocks. Original block-score top-k ranks max/logsumexp summaries; original top-k-from-score alone does not construct the required importance tensor.'
        if op=='CumsumFwdOp':
            semantic='source_api_blocked'
            reason='Original cumsum host sets BLOCK_T=next_power_of_2(2**18/(H*chunk_size)); full manifest rows need chunk_size>=4096, making BLOCK_T<chunk_size and native N_CHUNKS=0. Fixing block scheduling or adding cross-chunk prefix arithmetic is outside scope.'
        if op in ('MoEPrePermuteFwdOp','MoEPostPermuteFwdOp','MoEPermuteAlignFwdOp') and not own:
            semantic='deployment_blocked'
            reason='Own distributed dispatch/combine has multi-rank topology requirements on this single-NPU host; no single-rank equivalent standalone original API found. This is a deployment/topology limitation, not a claim that the math is absent.'
        if op in ('CausalConv1dFwdOp','CausalConv1dUpdateFwdOp','FusedQKNormRoPEFwdOp') and op in missing:
            semantic='supported_missing_workload_factory'
            reason='Direct original SGL API exists in causal_conv1d / fused_split_qk_norm+RoPE; shared inventory reports missing workload factory, so no authentic Case or numerical test can be fabricated.'
        if op in reference_blocked:
            semantic='reference_blocked'
            reason='The original shared workload explicitly declares ReferenceInfeasible: W4A8 packed-weight/repack and dequantization semantics await the user contract. No oracle, candidate failure, or mathematical impossibility is fabricated.'
            evidence.append(source('TileOPs/workloads/ascend_extended_transform_quant.py'))
        op_rows=[]
        for c in [c for c in inv['cases'] if c['op']==op]:
            row={'op':op,'case_id':c['case_id'],'library':'sgl-kernel-npu','status':'supported_untested' if own else semantic,
                 'semantic_status':semantic,'reason':reason,'adapter_path':adapter,'adapter_line':fnline,'source_evidence':evidence,'numerical_receipt':None,'validation_origin':None,
                 'workload_factory_available':op not in missing}
            row['reference_status']='explicit_reference_infeasible' if op in reference_blocked else 'available'
            if op=='PagedKVCacheGatherFwdOp' and (EVIDENCE/'unidex_cpu_build_initial.json').exists():
                row['deployment_receipt']=str((EVIDENCE/'unidex_cpu_build_initial.json').relative_to(ROOT))
            key=(op,c['case_id'])
            restricted=restriction(c) if own else None
            if restricted:
                row.update(status='unsupported_case',reason=restricted,case_semantic_status='source_parameter_restriction')
            if op in missing:
                row['status']='missing_workload_factory'
            elif key in old:
                hist=old[key]
                row['historical_record']=hist
                receipt=hist.get('device_receipt') or hist.get('last_receipt')
                if hist['status']=='verified' and receipt and (ROOT/receipt).is_file():
                    hr=json.loads((ROOT/receipt).read_text())
                    mismatches=[p for p,h in hr.get('source_sha256',{}).items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
                    row.update(status='verified_historical' if not mismatches else 'historical_source_drift',reason=hist.get('reason',''),numerical_receipt=receipt,validation_origin='historical_classic_matrix',historical_source_mismatches=mismatches)
                elif hist['status']=='unsupported_within_scope':
                    row.update(status='unsupported_within_scope',reason=hist['reason'])
                else:
                    row.update(status='historical_failure',reason=hist.get('reason',''),numerical_receipt=receipt,validation_origin='historical_classic_matrix')
            if op in ('Conv2dFwdOp','TopKMaskFwdOp') and own and row['status'] in ('unsupported_within_scope','historical_failure'):
                row.update(status='supported_untested',reason='New isolated adapter broadens the historical route; no old failure or old API exclusion is reused as a current mathematical conclusion.',validation_origin=None)
            if key in incomplete_attempts and key not in receipts:
                attempt=incomplete_attempts[key]
                code=attempt['exit_code']
                row.update(status='execution_timeout_without_receipt' if code==124 else 'device_reservation_blocked',reason=f"Guard exit {code}; no completed numerical receipt for this case. See exact queue/log; no numeric result inferred.",attempt_log=attempt['log'],attempt_exit_code=code,validation_origin='this_worker')
            if key in receipts:
                f,r=receipts[key]
                status=r['status']
                if status=='verified' and r.get('attribution_complete') and r.get('verification',{}).get('checked_outputs',0)>0 and r.get('verification',{}).get('checked_outputs')==r.get('verification',{}).get('total_outputs'):
                    status='verified'
                elif status=='verified':
                    status='attribution_failed' if not r.get('attribution_complete') else 'output_contract_failed'
                elif status=='unsupported':status='unsupported_case'
                elif status=='failed':
                    import re
                    frames=re.findall(r'File "([^"]+)"',r.get('traceback',''))
                    last=frames[-1] if frames else ''
                    status='numerical_failed' if r.get('error_type')=='AssertionError' and r.get('stage') in ('verify','edge_check') and ('/workloads/' in last or '/torch/testing/' in last or last.endswith('ascend_classic_check.py')) else 'deployment_or_runtime_blocked'
                    if r.get('stage')=='verify' and r.get('candidate_calls',{}).get('entered')==0:
                        status='reference_or_verifier_blocked'
                    if 'support feature dim' in r.get('error',''):
                        status='source_api_blocked'
                    if 'ub overflow' in r.get('error','').lower():
                        status='compiler_resource_blocked'
                    if 'reference establishes' in r.get('error',''):
                        status='output_contract_failed'
                if status=='verified':
                    drift=sorted({path for name in ('source_sha256','cpu_prepared_source_sha256','oracle_source_sha256','native_trace_source_sha256')
                                  for path,h in r.get(name,{}).items()
                                  if not (ROOT/path).exists() or sha(ROOT/path)!=h})
                    row['current_source_mismatches']=drift
                    if drift:status='verified_prior_adapter_revision'
                    if f.parent.name in contaminated_batches:status='device_run_contaminated'
                if r.get('cpu_prepared_artifact') or r.get('input_generation_device') == 'cpu':
                    status='held_for_original_NPU_input_rerun'
                row.update(status=status,reason=receipt_reason(r),numerical_receipt=str(f.relative_to(ROOT)),validation_origin='this_worker',api=r.get('api'),adaptation=r.get('adaptation'),source_sha256=r.get('source_sha256'),verification=r.get('verification'),runtime_original_calls=r.get('runtime_original_calls'),runtime_native_calls=r.get('runtime_native_calls'),attribution_complete=r.get('attribution_complete'))
                row.update(native_trace_schema=r.get('native_trace_schema',1),native_kernel_launches=r.get('native_kernel_launches',[]),
                           candidate_calls=r.get('candidate_calls'),input_preservation_requested=r.get('input_preservation_requested',True),
                           reference_device=r.get('reference_device'),cpu_prepared_artifact=r.get('cpu_prepared_artifact'),
                           cpu_prepared_source_sha256=r.get('cpu_prepared_source_sha256'),oracle_source_sha256=r.get('oracle_source_sha256'),
                           native_trace_strength='direct_candidate_launch' if r.get('native_trace_schema',1)>=2 else 'historical_vendor_python_call')
                alternate=PENDING_ALTERNATIVES.get(key)
                alternate_path=f'TileOPs/benchmarks/ascend_all_pairs/sgl/{alternate}' if alternate else None
                if alternate and r['status']=='failed' and alternate_path not in r.get('source_sha256',{}):
                    row['prior_attempt_status']=status
                    row['receipt_adapter_revision']='prior_public_api_route'
                    row['pending_adapter_source']=source(alternate_path)
                    row.update(status='supported_untested',reason=f'Previous original API route failed ({status}); its full receipt and attribution remain attached. A different original API route in {alternate} is implemented but has not been device-tested. Previous diagnostic: '+receipt_reason(r))
            if op in reference_blocked:
                row.update(status='reference_blocked',reason=reason)
            op_rows.append(row);rows.append(row)
        counts=Counter(r['status'] for r in op_rows)
        operators.append({'op':op,'family':family,'semantic_status':semantic,'reason':reason,'adapter_path':adapter,'adapter_line':fnline,'source_evidence':evidence,'case_count':len(op_rows),'case_status_counts':dict(counts),'workload_factory_available':op not in missing})
    summary={'operator_count':len(operators),'case_count':len(rows),'registered_candidate_ops':len(PROVIDERS),'implemented_new_ops':sorted(set(PROVIDERS)-{r['op'] for r in classic['rows']}),'verified_new_ops':sorted({r['op'] for r in rows if r['status']=='verified'}),'new_verified_cases':sum(r['status']=='verified' for r in rows),'historical_verified_cases':sum(r['status']=='verified_historical' for r in rows),'case_status_counts':dict(Counter(r['status'] for r in rows)),'missing_workload_factory_ops':sorted(missing),'reference_blocked_ops':sorted(reference_blocked),'prior_attempt_status_counts':dict(Counter(r['prior_attempt_status'] for r in rows if 'prior_attempt_status' in r))}
    obj={'schema':'sgl-all-pairs/v1','schema_description':{'operators':'Exactly one source-semantic conclusion per inventory operator.','cases':'Exactly one row per immutable inventory manifest case; verified requires a complete numeric receipt and recorded original SGL runtime calls.','verified_historical':'Preserved classic receipt, current recorded source hashes match; not rerun here.','supported_untested':'Implemented original-library adapter, no numeric claim.','reference_blocked':'Original shared ReferenceInfeasible declaration; mathematical acceptance awaits the missing user contract, with no fabricated reference.', 'unsupported_within_scope':'No equivalent complete original API under allowed adaptations.','source_api_blocked':'Existing source operation cannot express requested shape with unchanged public host schedule.','compiler_resource_blocked':'Real original-kernel compilation exceeded device resource limits; exact diagnostic retained.','output_contract_failed':'Candidate returned a different number/structure of outputs than the unchanged reference requires.','deployment_blocked':'Source route is unavailable in the installed binary or requires unsupported hardware topology; not a mathematical impossibility.','deployment_or_runtime_blocked':'Real import/build/device/reference exception; see exact receipt.','numerical_failed':'Real original tolerance/input/control assertion failed.','reference_or_verifier_blocked':'Original oracle/verifier failed before the candidate run was entered; no candidate numerical conclusion is inferred.','missing_workload_factory':'Manifest case exists but shared Case factory is absent; cannot test honestly.','execution_timeout_without_receipt':'Guard stopped a started case before complete receipt; no numerical claim.','device_reservation_blocked':'Guard could not grant a clean run; exact exit code and log retained.','device_run_contaminated':'Guard observed another device user; candidate acceptance revoked for that batch.','verified_prior_adapter_revision':'Numeric receipt passed, but one recorded source fingerprint changed afterward.','unsupported_case':'Specific dtype/parameter contract has no equivalent unchanged adapter; reason is static source restriction or a real UnsupportedCase receipt, distinguished by validation_origin.'},'updated_at':datetime.now(timezone.utc).isoformat(),'inventory_snapshot':str(invpath.relative_to(ROOT)),'inventory_sha256':sha(invpath),'summary':summary,'operators':operators,'cases':rows}
    (EVIDENCE/'coverage.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    p=EVIDENCE/'progress.json'
    progress=json.loads(p.read_text()) if p.exists() else {}
    progress.update(summary,verified_new_cases=summary['new_verified_cases'],candidate_op_count=len(PROVIDERS),updated_at=obj['updated_at'],coverage_path=str((EVIDENCE/'coverage.json').relative_to(ROOT)),semantic_inventory_complete=len(operators)==205,device_validation_complete=not any(r['status'] in ('supported_untested','execution_timeout_without_receipt','device_reservation_blocked','device_run_contaminated','verified_prior_adapter_revision','missing_workload_factory','historical_source_drift','historical_failure','reference_blocked') for r in rows))
    p.write_text(json.dumps(progress,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
