"""Reproducible CPU-only coverage and source-contract audit for all manifest ops."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from . import PROVIDERS
from . import native
from .deployment import probe_deployment
from .metadata import inventory, manifest, parameters
from .plans import assess

HERE=Path(__file__).resolve().parent
WORKSPACE=native.WORKSPACE
VENDOR=native.VENDOR
ELEMENTWISE='testing/python/language/test_tilelang_ascend_language_elementwise.py'
API='tilelang/language/ascend_tile.py'
UNARY='testing/python/base/unary_op.py'
REDUCE=ELEMENTWISE
GQA='examples/gqa_fwd_varlen/gqa_fwd_varlen.py'
SPARSE='examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py'
SMASK='examples/sparse_flash_attention/example_sparse_flash_attn_mask.py'
SMLA='examples/deepseek_v4/sparse_flash_mla/sparse_flash_mla_kernel.py'
GDN='examples/linear_attention_and_rnn/gdn_full.py'
GH='examples/linear_attention_and_rnn/gdn/gdn_chunk_h.py'
FUSED_GDN='examples/fused_sigmoid_gating_delta_rule/fused_sigmoid_gating_delta_rule_varlen.py'
MOE='examples/fusedmoe/example_fusedmoe.py'
QCAST='examples/tile_kernels/quant/per_block_cast_lossless_kernel.py'
DQUANT='examples/dynamic_quant/dynamic_quant.py'
FQUANT='examples/add_rms_norm_dynamic_quant/add_rms_norm_dynamic_quant.py'
TOPK='examples/topk_selector/example_topk_selector.py'
NORM='examples/cann-bench/group_norm/example_group_norm.py'
RMS='examples/normalization/rms_norm.py'
ROPE='examples/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py'
CUMMIN='examples/cann-bench/cummin/example_cummin.py'

# These are scoped source-contract conclusions, not claims that the accelerator
# or language could never implement the missing operator with a new kernel.
UNSUPPORTED={}


def reject(ops,reason,*sources,kind='missing_original_semantics',status='unsupported'):
    for op in ops.split():
        assert op not in UNSUPPORTED,op
        UNSUPPORTED[op]=dict(reason=reason,source_specs=list(sources),rejection_kind=kind,status=status)


reject('AdaLayerNormFwdOp AdaLayerNormZeroFwdOp',
       'Original affine GroupNorm/LayerNorm does not take per-row adaptive scale/shift or the additional zero gate. Supplying those as torch arithmetic would replace the requested epilogue.',(NORM,'group_norm'),('examples/normalization/layer_norm.py','layer_norm'))
reject('BatchNormFwdOp BatchNormBwdOp',
       'GroupNorm reduces independent groups; no original batch-statistics running-mean/variance update or complete backward gradient/statistic output API was found. GroupNorm output alone does not satisfy the full manifest.',(NORM,'group_norm'))
reject('FusedAddLayerNormFwdOp FusedAddRMSNormFwdOp',
       'Inspected fused add+RMSNorm kernel returns quantized codes, residual sum and quantizer scale, not unquantized affine normalized output. Unweighted standalone norm lacks the complete fused output contract.',(FQUANT,'_kernel_impl'),(RMS,'_rms_norm'))
reject('FusedAddRMSNormPerTokenQuantFwdOp',
       'Original fused quantizer returns INT8 only and forces amax>=1e-12; manifest requires FP8 or INT8 plus exact scale (zero for a zero normalized row) and optional scale upper bound. The updated FP32-residual oracle was inspected; no residual-rounding mismatch is asserted.',(FQUANT,'_kernel_impl'),(FQUANT,'golden_add_rms_norm_dynamic_quant'),kind='quantizer_contract_mismatch')
reject('FusedQKNormRoPEFwdOp',
       'Original xLLM fused kernel validates a fixed head/rotary specialization and mRoPE cache representation, while manifest needs 128-wide heads, configurable 64/128 rotary prefix, affine Q/K norms, unchanged V and complete packed mutation. Adapting cache layout does not supply a missing normalization specialization.',('examples/xllm_kernels/split_qkv_rmsnorm_mrope.py','_validate_head_spec'),('examples/xllm_kernels/split_qkv_rmsnorm_mrope.py','build_split_qkv_rmsnorm_mrope_kernel'))
reject('AdaptiveAvgPool2dFwdOp AvgPool1dFwdOp AvgPool2dFwdOp AvgPool3dFwdOp MeanPoolingFwdOp',
       'Original row kernels expose sum/min/max, not an average with each manifest window divisor, padding-count/divisor override or adaptive-window normalization. A window layout alone cannot turn a sum into the required average.',(REDUCE,'reduce_runtime_semantics_kernel'))
reject('AdaptiveMaxPool2dIndicesFwdOp MaxPool1dIndicesFwdOp MaxPool2dIndicesFwdOp MaxPool3dIndicesFwdOp',
       'Original max reduction returns values only. Manifest also requires exact first-max spatial indices, including dilation/padding/ties; the inspected top-k API has no documented equivalent index-tie contract.',(REDUCE,'reduce_runtime_semantics_kernel'),(TOPK,'simple_topk_selector'),kind='incomplete_output_contract')

ELEMENTWISE_MISSING={
'AlibiFwdOp':'ALiBi requires per-head slopes and relative-position bias generation; no original complete ALiBi builder was found.',
'CeilFwdOp':'Nearest-integer round is not ceiling; the original round API has no selectable ceil mode.',
'FloorFwdOp':'Nearest-integer round is not floor; the original round API has no selectable floor mode.',
'TruncFwdOp':'Nearest-integer round is not truncation toward zero; the original round API has no selectable truncation mode.',
'ClampTensorFwdOp':'The original clamp builder accepts scalar bounds, not broadcast tensor min/max bounds.',
'DropoutFwdOp':'No original Bernoulli mask/RNG and rescaling API matching manifest dropout was found.',
'EluFwdOp':'Native ReLU/LeakyReLU does not implement ELU exponential negative branch and alpha.',
'ErfFwdOp':'No original erf tiled builder or tile intrinsic is present in the inspected owned corpus.',
'Expm1FwdOp':'Exp is not expm1 at cancellation-sensitive inputs; no original expm1 API was found.',
'FloorDivideFwdOp':'Original vec_div performs true division and lacks floor integer quotient semantics.',
'GeluAndMulFwdOp':'Original gelu_mul is explicitly tanh-approximate; manifest requires erf-based exact GELU.',
'GeluFwdOp':'Manifest cases request approximate=none; inspected gelu_mul/gelu_grad implement tanh-approximate formulas, not exact erf GELU.',
'HardsigmoidFwdOp':'No original clipped affine hard-sigmoid kernel was found; sigmoid is different mathematics.',
'HardswishFwdOp':'No original hard-swish kernel was found; native SiLU is a different activation.',
'IsfiniteFwdOp':'Native numeric comparison/bit packing is not a complete finite/inf/NaN classification API.',
'IsinfFwdOp':'No original two-sign infinity classification kernel was found.',
'LerpScalarFwdOp':'No original scalar-weight lerp builder was found; binary add/mul alone is not this complete original operation.',
'LerpTensorFwdOp':'No original broadcast tensor-weight interpolation API was found.',
'Log1pFwdOp':'Native log lacks the log1p input law and cancellation behavior.',
'MishFwdOp':'No original complete softplus/tanh/multiply Mish API was found.',
'NanToNumFwdOp':'No original configurable NaN/+inf/-inf replacement API was found.',
'PreluFwdOp':'Original LeakyReLU exposes one scalar slope, not a learned per-channel tensor slope.',
'RemainderFwdOp':'Original division lacks signed remainder/floor-mod semantics.',
'SeluFwdOp':'No original scaled ELU complete API was found.',
'SignFwdOp':'No original three-way sign kernel with manifest zero/NaN behavior was found.',
'SinusoidalFwdOp':'No original complete sinusoidal embedding builder with manifest placement/frequency law was found; rotary transforms existing data instead.',
'SoftplusFwdOp':'No standalone original softplus API matching beta/threshold was found; fused recurrent softplus is inseparable from its state update.',
}
for op,reason in ELEMENTWISE_MISSING.items():
    sources=[(API,None),(UNARY,'make_unary_kernel')]
    if op in ('GeluFwdOp','GeluAndMulFwdOp'):sources=[('examples/activation/gelu_mul.py','gelu_mul')]
    if op=='ClampTensorFwdOp':sources=[(ELEMENTWISE,'clamp')]
    reject(op,reason,*sources)

REDUCTION_MISSING={
'ArgmaxFwdOp':'Max kernel has no index output; top-k does not establish first-index tie semantics. The cumulative-min API provides the opposite order statistic.',
'CountNonzeroFwdOp':'No original nonzero predicate plus exact count reduction API was found.',
'CumprodFwdOp':'Original scan is cumulative minimum, not prefix product.',
'CumsumFwdOp':'Original standalone scan is cumulative minimum. GDN chunk-cumsum is chunk-reset/gating-specific and does not expose the complete manifest prefix sum.',
'LogSoftmaxFwdOp':'Original online softmax returns probabilities only, not log probabilities.',
'LogSumExpFwdOp':'Original softmax keeps max/normalizer internally but does not return the required complete logsumexp reduction.',
'MeanFwdOp':'Original sum reduction has no divisor parameter or mean output.',
'ProdFwdOp':'Original reduction dispatch only supports sum/min/max, not product.',
'StdFwdOp':'No original variance/std reduction with correction and complete shape semantics was found.',
'VarFwdOp':'No original variance reduction with degrees-of-freedom correction was found.',
'VarMeanFwdOp':'No original complete two-output variance and mean API was found.',
'VectorNormFwdOp':'Unweighted RMSNorm returns normalized elements, not a vector-norm reduction or configurable p norm.',
}
for op,reason in REDUCTION_MISSING.items():
    refs=[(REDUCE,'reduce_runtime_semantics_kernel')]
    if op in ('CumsumFwdOp','CumprodFwdOp','ArgmaxFwdOp'):refs.append((CUMMIN,'cummin_vec_ker'))
    if op in ('LogSoftmaxFwdOp','LogSumExpFwdOp'):refs.append(('examples/softmax/example_online_softmax.py','online_softmax'))
    reject(op,reason,*refs)
reject('BmmFP8FwdOp GemmFP8FwdOp MoEGroupedGemmFP8FwdOp',
       'Original raw FP8 GEMM has no manifest per-block activation/weight dequantization scales or scaled grouped-output contract. The per-block cast example forces FP32 output and is not an FP8 scaled GEMM substitute.',('examples/gemm/example_gemm_fp8_pto.py','fp8_matmul'),(QCAST,'per_block_cast_lossless'),kind='missing_scale_semantics')
reject('GemmW4A16FwdOp GemmW4A8FwdOp INT4QuantPerGroupFwdOp',
       'No owned original packed 4-bit kernel with manifest signed nibble/zero-point/group-scale semantics was found. INT8 and raw FP8 kernels are not 4-bit packed implementations.',('examples/deepseek_v4/int8_gemm.py','int8_gemm_kernel_corrected'),(QCAST,'per_block_cast_lossless'))
reject('INT8QuantPerTensorFwdOp INT8QuantPerChannelFwdOp INT8QuantPerBlockFwdOp',
       'Original dynamic quant uses max(amax,1e-12)/127 for zero rows; manifest symmetric INT8 oracle uses scale=1 when amax=0. Reshaping changes axes but cannot repair returned scales without changing quantizer mathematics.',(DQUANT,'_dynamic_quant_kernel'),('examples/deepseek_v4/act_quant.py','act_quant_kernel_int8_optimized'),kind='quantizer_contract_mismatch')
reject('FP8QuantFwdOp FP8QuantPerBlockFwdOp',
       'The inspected per_block_cast_lossless host entry explicitly selects FP32 output even for requested low-precision formats; no complete native E4M3 quantized-code plus manifest-scale API was found. An FP32 output cannot stand in for FP8 codes.',(QCAST,'per_block_cast_lossless'),kind='incomplete_output_contract')
reject('SmoothQuantFwdOp',
       'Dynamic quant lacks the manifest pre-quantization channel smoothing tensor and its output scale contract; external torch multiplication would substitute for the missing fused mathematics.',(DQUANT,'_dynamic_quant_kernel'))

reject('SamplingFromProbsFwdOp ChainSpeculativeSamplingFwdOp',
       'Original top-k selects deterministic indices; it does not implement probabilistic sampling, random uniforms, speculative accept/reject or residual-distribution resampling outputs.',(TOPK,'simple_topk_selector'))
reject('TopKMaskFwdOp TopPMaskFwdOp MinPMaskFwdOp TopKTopPMaskFwdOp',
       'Original top-k returns index lists only. Manifest requests complete tie-aware masked/renormalized probability tensors, cumulative top-p or relative min-p thresholds; no matching original mask/normalization API was found.',(TOPK,'simple_topk_selector'),('examples/tile_kernels/moe/moe_topk_gate.py','topk_gate'))
reject('TopKSelectFwdOp',
       'Closest native selector allocates score_accum[N] plus sorting workspace in UB and documents N around 8192 for 192KB UB; all manifest N are 16384/65536. Complete window/sentinel semantics also need an adapter. No device/capacity experiment was permitted and no kernel tiling rewrite is allowed; this is a deployment/capacity gap, not mathematical impossibility.',(TOPK,'simple_topk_selector'),kind='native_capacity_and_adapter_gap',status='deployment_blocked')

reject('DSADecodeWithKVCacheFwdOp',
       'Native dynamic sparse MLA consumes every listed index without sentinel masking/deduplication. Its mask variant compares indices to local query row s_i, whereas manifest uses q_start_index_s, compressed stride and a KV-length sentinel. These complete selection semantics are not provided by the inspected original APIs; no existing-shape difference alone is used as rejection.',(SPARSE,'sparse_attention_fwd'),(SMASK,'sparse_attention_fwd'),kind='selection_contract_mismatch')
reject('DSAPagedFwdOp',
       'Manifest consumes packed uint8 KV records with quantization metadata and returns both O and LSE. Inspected sparse attention uses homogeneous FP16/BF16 KV tensors and returns O only; layout reshaping cannot decode missing quantizer mathematics or expose LSE.',(SPARSE,'sparse_attention_fwd'),(SMLA,'build_sparse_flash_mla'),kind='incomplete_output_contract')
reject('MLAPagedFwdOp MLAVarlenFwdOp',
       'Manifest requires complete O plus LSE and (varlen) separately supplied K/V with unequal query/value widths. Audited sparse MLA returns O only and uses the leading KV latent fields as V. The native GQA route also returns no LSE and has fixed equal 128-wide Q/K/V layouts.',(SPARSE,'sparse_attention_fwd'),(GQA,'flashattn'),kind='incomplete_output_contract')
reject('GQAPrefillPagedWithKVCacheFwdOp',
       'Inspected native GQA does not own cache append/writeback, quantized-cache scales or the fused RoPE required by these prefill cases. Returning only attention output omits declared full cache effects; no torch cache-only implementation is credited.',(GQA,'flashattn'),(ROPE,'rope_kernel'),kind='incomplete_state_contract')
reject('PagedKVCacheWriteFwdOp MLAKVCacheWriteFwdOp PagedKVCacheGatherFwdOp',
       'No original standalone cache gather/scatter/write API was found. Native attention reads caches as part of attention but cannot return the requested complete cache/gather result. Implementing the requested operation solely with torch indexing would have no actual target-library operation.',(SMLA,'build_sparse_flash_mla'),(GQA,'flashattn'),kind='no_owned_cache_primitive')
reject('MergeAttentionStatesFwdOp',
       'Online softmax combines internal accumulators but exposes no original API taking two normalized outputs plus their LSE states and returning both merged output and merged LSE.',('examples/softmax/example_online_softmax.py','online_softmax'),(GQA,'flashattn'),kind='incomplete_output_contract')
reject('FP8LightningIndexerFwdOp',
       'Original lightning indexer performs top-k index selection and its low-precision templates use FP16/BF16; manifest requests the full weighted-ReLU logits tensor with FP8 block scales, query-wise start/end windows and clean_logits behavior. Intermediate QK scratch or selected scores are not the complete returned logits contract.',('examples/lightning_indexer/example_lightning_indexer.py','indexer'),('examples/deepseek_v4/lightning_indexer.py','make_lightning_indexer_kernel'),kind='incomplete_output_contract')
reject('NSACompressedVarlenFwdOp',
       'Manifest compressed attention requires both O and per-head LSE under chunk visibility. Original GQA/sparse APIs inspected return O only and do not expose compressed chunk normalizers.',(GQA,'flashattn'),(SPARSE,'sparse_attention_fwd'),kind='incomplete_output_contract')
reject('NSATopKVarlenFwdOp',
       'Manifest selects blocks after grouped-head score aggregation and packed chunk visibility. Original token top-k and lightning indexer have different score aggregation and masking contracts.',(TOPK,'simple_topk_selector'),('examples/lightning_indexer/example_lightning_indexer.py','indexer'))
reject('NSAVarlenFwdOp',
       'Manifest supplies independent K and V, selected block counts and packed-sequence visibility. Inspected sparse MLA reuses latent K as V and reads a fixed number of selected indices; its mask variant has local-row causal semantics, not the complete counted-block/offset contract.',(SPARSE,'sparse_attention_fwd'),(SMASK,'sparse_attention_fwd'))

reject('GDNFwdOp DeltaNetFwdOp DeltaNetChunkFwdOp DeltaNetRecurrentFwdOp',
       'Chunk GDN entry discards final state; underlying chunk-state kernel stores state in input dtype, initializes its FP32 running state to zero, and lacks general FP32 initial-state semantics. Fused recurrent kernel has built-in softplus/sigmoid decay and input-dtype state, different from manifest precomputed gates, optional transforms and complete FP32 state. No state output is silently omitted.',(GDN,'kernel_chunk_gdn'),(GH,'chunk_h_ker'),(FUSED_GDN,'kernel'),kind='incomplete_state_contract')
reject('DeltaNetChunkBwdOp GLAChunkBwdOp',
       'No original complete DeltaNet/GLA recurrent backward API with all manifest input/state gradients was found. GDN forward kernels do not satisfy a backward contract.',(GDN,'kernel_chunk_gdn'),(FUSED_GDN,'kernel'))
reject('GLAFwdOp GLAChunkFwdOp GLARecurrentFwdOp KDAFwdOp',
       'Original GDN uses scalar-head decay plus delta correction. Manifest GLA/KDA requires feature-dependent gating and different recurrence/state equations; reshaping scalar gates cannot implement those mathematical laws.',(GDN,'kernel_chunk_gdn'),(FUSED_GDN,'kernel'),kind='recurrence_math_mismatch')
reject('SSDChunkCouplingFwdOp SSDChunkStateFwdOp SSDStatePassingFwdOp SSDChunkScanFwdOp SSDRecurrentFwdOp Mamba2FwdOp SelectiveScanFwdOp',
       'Inspected recurrent kernels implement gated delta-rule attention, not SSD/Mamba discretization, chunk coupling/state passing or selective-scan B/C/D/z equations. They also lack the complete declared intermediate/final-state outputs. No torch recurrence substitute is registered.',(GDN,'kernel_chunk_gdn'),(FUSED_GDN,'kernel'),kind='recurrence_math_mismatch')
reject('SSDChunkCumsumFwdOp',
       'GDN chunk-cumsum only accumulates supplied gates; SSD chunk-cumsum also applies dt bias/softplus/clamping and A-dependent transforms and returns multiple tensors. Native cumulative-min is a different scan.',('examples/linear_attention_and_rnn/gdn/gdn_chunk_cumsum.py',None),(CUMMIN,'cummin_vec_ker'))

reject('FusedMoEExpertsFwdOp FusedMoEFwdOp FusedMoESharedExpertFwdOp IndexedExpertMLPFwdOp MoEExpertMLPFwdOp SharedExpertMLPFwdOp',
       'Vendor fusedmoe host wrappers explicitly compute SiLU, gate product, routing weights and combinations with torch/F, around native GEMMs. Crediting those wrappers would violate owned-native-call requirements for complete MLP/MoE. Plain expert GEMM is separately registered.',(MOE,'shared_expert_kernel'),(MOE,'routed_expert_kernel'),kind='dependency_math_fallback')
reject('FusedTopKFwdOp',
       'Original topk_gate returns selected indices from already computed scores, not manifest scoring sigmoid/softmax, correction bias, expert-group choice, renormalized weights and routing scale outputs.',('examples/tile_kernels/moe/moe_topk_gate.py','topk_gate'),('examples/tile_kernels/moe/moe_topk_sum_and_topk_group_idx.py',None),kind='incomplete_output_contract')
reject('MoEPrePermuteFwdOp',
       'Original MoeTokenPermute returns permuted tokens and inverse map but no complete expert prefix-end metadata required as the third manifest output. Recomputing that result with torch bincount/cumsum is excluded.',('examples/moe_token_permute/moe_token_permute.py','MoeTokenPermute'),kind='incomplete_output_contract')
reject('MoEPermuteAlignFwdOp',
       'Original token permute provides tight reordered tokens/inverse indices, not stable padded route ids, per-block expert ids and total padded count. The complete alignment metadata contract is absent.',('examples/moe_token_permute/moe_token_permute.py','MoeTokenPermute'),kind='incomplete_output_contract')
reject('EngramDecodeFwdOp EngramGateConvFwdOp EngramGateConvBwdOp',
       'No original complete hash/ngram lookup plus embedding/gated convolution (or its backward) was found. Width-4 causal depthwise convolution alone omits Engram lookup, gating, state and gradient contracts.',('examples/causal_conv1d/causal_conv1d_decode.py','build_causal_conv1d_decode_kernel'))
reject('FFTC2CFwdOp HadamardTransformFwdOp',
       'No owned original FFT/complex butterfly or Hadamard transform API was found in the 309-file kernel/language corpus. Generic GEMM/elementwise builders are not credited as an absent complete transform.',(API,None),(ELEMENTWISE,'vec_add_developer'))
reject('MHCPreFwdOp',
       'hc_split_sinkhorn starts from projected mixes and computes split/gating/Sinkhorn; head_compute_mix is another partial stage. Neither original API provides complete manifest input normalization, projection and all mixing/output tensors.',('examples/deepseek_v4/hc_split_sinkhorn.py','hc_split_sinkhorn'),('examples/tile_kernels/mhc/head_compute_mix_kernel.py','_mhc_head_compute_mix_fwd'),kind='incomplete_output_contract')


@lru_cache(None)
def source_record(relative,symbol=None):
    path=VENDOR/relative;text=path.read_text();tree=ast.parse(text)
    nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (symbol is None or n.name==symbol)]
    if symbol is not None and not nodes:raise ValueError(f'Missing evidence definition {relative}:{symbol}')
    first=min(nodes,key=lambda n:n.lineno) if nodes else None
    return {'file':str(path.relative_to(WORKSPACE)),'line':first.lineno if first else 1,
            'end_line':first.end_lineno if symbol and first else len(text.splitlines()),
            'symbol':symbol,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def factory_audit():
    """Static presence only, no workload imports and no claim of runtime readiness."""
    sources=defaultdict(list)
    for path in (WORKSPACE/'TileOPs/benchmarks/_cases').glob('*.py'):
        text=path.read_text();tree=ast.parse(text)
        for n in ast.walk(tree):
            if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.endswith(('FwdOp','BwdOp')):
                sources[n.value].append({'file':str(path.relative_to(WORKSPACE)),'line':n.lineno,
                                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return sources


def main():
    inv=inventory();now=datetime.now(timezone.utc).isoformat()
    own_ops={row['op'] for row in inv['operators']}
    assert set(UNSUPPORTED)|set(PROVIDERS)==own_ops,(own_ops-(set(UNSUPPORTED)|set(PROVIDERS)))
    assert not set(UNSUPPORTED)&set(PROVIDERS)
    static=json.loads((HERE/'static_validation.json').read_text())
    receipts=json.loads((HERE/'static_case_receipts.json').read_text())['cases']
    by_case={(r['op'],r['case_id']):r for r in receipts}
    factories=factory_audit()
    source_catalog=json.loads((HERE/'source_catalog.json').read_text())
    owned_catalog=[r for r in source_catalog if '/3rdparty/' not in r['file']]
    deployment=probe_deployment(VENDOR, native.CACHE)
    deployment.update({
        'loader_source':source_record('tilelang/libinfo.py','get_dll_directories'),
        'tvm_loader_source':source_record('3rdparty/tvm/python/tvm/_ffi/libinfo.py','get_dll_directories'),
        'pto_header':{'file':'ascend_baselines/tilelang-ascend/3rdparty/pto-isa/include/pto/pto-inst.hpp',
                      'exists':(VENDOR/'3rdparty/pto-isa/include/pto/pto-inst.hpp').is_file()},
    })
    results=[];grouped=defaultdict(list)
    for row in inv['cases']:
        op=row['op'];p,resolved=parameters(row);decision=assess(op,p)
        rec=by_case.get((op,row['case_id']))
        if decision.supported:
            assert rec is not None,(op,row['case_id'])
            status='supported_unverified';reason=decision.reason;kind=None;sources=rec['sources']
            adapter='TileOPs/benchmarks/ascend_all_pairs/tla/adapters.py'
        else:
            audit=UNSUPPORTED.get(op)
            status=audit['status'] if audit else 'unsupported'
            reason=audit['reason'] if audit else decision.reason
            kind=audit['rejection_kind'] if audit else 'case_contract_restriction'
            sources=([source_record(*s) for s in audit['source_specs']] if audit else
                     next(r['sources'] for r in receipts if r['op']==op))
            adapter=None
        result={'op':op,'case_id':row['case_id'],'status':status,
            'reason':reason,'rejection_kind':kind,'provider_registered':op in PROVIDERS,
            'case_adapter_available':decision.supported,'adapter_path':adapter,
            'api':rec['api'] if rec else None,'adaptation':rec['adaptation'] if rec else None,
            'timing_mode':rec['timing_mode'] if rec else None,'sources':sources,
            'params':p,'resolved_adt_fields_from_manifest':resolved,
            'workload_factory':{'static_presence':bool(factories.get(op)),'sources':factories.get(op,[])},
            'deployment_status':deployment['status'] if decision.supported else 'not_attempted',
            'static_receipt_path':'TileOPs/benchmarks/ascend_all_pairs/tla/static_case_receipts.json' if rec else None,
            'static_receipt_key':[op,row['case_id']] if rec else None,
            'numerical_receipt_path':None,'input_protection_device_receipt':None,'native_ownership_device_receipt':None,
            'device_runs':0,'verified':False}
        results.append(result);grouped[op].append(result)
    operators=[]
    for row in inv['operators']:
        op=row['op'];items=grouped[op];counts=dict(Counter(r['status'] for r in items));path=WORKSPACE/'TileOPs'/row['manifest']
        text=path.read_text();line=next(i for i,s in enumerate(text.splitlines(),1) if s.startswith(op+':'))
        spec=manifest(op)
        operators.append(dict(op=op,category=row['category'],case_count=len(items),case_status_counts=counts,
            status='supported_unverified' if any(r['case_adapter_available'] for r in items) else next(iter(counts)),
            provider_registered=op in PROVIDERS,adapter_path='TileOPs/benchmarks/ascend_all_pairs/tla/adapters.py' if op in PROVIDERS else None,
            manifest={'file':str(path.relative_to(WORKSPACE)),'line':line,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
            declared_outputs=list(spec['signature']['outputs']),
            declared_mutations=[k for k,v in spec['signature'].get('inputs',{}).items() if v.get('mutated')],
            full_adapter_outputs=(['output','updated_conv_state'] if op.startswith('CausalConv1d') else list(spec['signature']['outputs'])) if op in PROVIDERS else None,
            source_evidence=items[0]['sources'],
            numerical_receipt_path=None,verified=False,
            notes=UNSUPPORTED[op]['reason'] if op in UNSUPPORTED else 'Per-case restrictions apply; full output metadata checked; actual numerical/ownership/input-preservation device receipts remain absent.'))
    missing_factory=[row['op'] for row in inv['operators'] if not factories.get(row['op'])]
    summary={'operators':len(operators),'manifest_cases':len(results),'registered_providers':len(PROVIDERS),
             'operator_status_counts':dict(Counter(r['status'] for r in operators)),
             'case_status_counts':dict(Counter(r['status'] for r in results)),
             'supported_cases_deployment_blocked':sum(r['case_adapter_available'] and deployment['status']=='deployment_blocked' for r in results),
             'numerical_failures':0,'verified_cases':0,'device_runs':0,
             'owned_source_files_catalogued':len(owned_catalog),'missing_factory_ops_static':missing_factory}
    coverage={'schema':'tilelang-ascend-all-pairs/v1','library':'tilelang-ascend','generated_utc':now,'owner':'allpairs-remote/tla',
        'schema_notes':{'supported_unverified':'Concrete original-native adapter exists and CPU metadata contract checks pass; no numerical/device claim.',
            'unsupported':'No equivalent complete path registered under the permitted unchanged-kernel adapter scope; scoped source-contract reason given, not a hardware impossibility claim.',
            'deployment_blocked':'Capacity/deployment gap is distinct from mathematical rejection; may lack a registered adapter.',
            'numerical_failed':'Reserved for actual device numerical failure receipts; never produced by this CPU-only task.',
            'deployment_status':'Orthogonal to semantic status: deployment_blocked records absent/incompatible files; unverified only records compatible ELF files, never runtime success.',
            'numerical_receipt_path':'null means no numerical measurement, never static-pass-as-verified.',
            'source_evidence':'Vendor original file/line/hash; unsupported absence conclusions also refer to the complete owned-source catalogue.',
            'inventory':'All current 2062 rows are retained, including the initial requested 1982 and 80 newly enumerated rows.'},
        'inventory_snapshot':{'file':'TileOPs/benchmarks/ascend_all_pairs/tla/inventory_snapshot.json',
            'sha256':hashlib.sha256((HERE/'inventory_snapshot.json').read_bytes()).hexdigest(),
            'original_requested_cases':1982,'snapshot_cases':len(results),'snapshot_stale_missing_factory_list':inv.get('missing_factory_ops')},
        'summary':summary,'deployment':deployment,'static_validation_path':'TileOPs/benchmarks/ascend_all_pairs/tla/static_validation.json',
        'source_catalog_path':'TileOPs/benchmarks/ascend_all_pairs/tla/source_catalog.json',
        'source_catalog_third_party_entries_not_credited':len(source_catalog)-len(owned_catalog),
        'harness_notes':['Shared inventory changed from 1982 to 2062 cases during this task.',
            'Opaque ADT repr fields are resolved from matching YAML workload labels/defaults in metadata.py; original inventory snapshot remains unchanged.',
            'Workload factory presence is AST evidence only; importing the real registry was forbidden. The snapshot missing_factory_ops list is stale; use current static list in summary.',
            'Causal factories return (output, updated_state) to verify declared mutation while adapters preserve original input tensors.',
            'GQABwd manifest uses base-2 LSE and q-zero controls. Adapter recomputes original native forward O/natural-log LSE each run, charging all cost; no oracle modification.',
            'No original manifest, reference, tolerance, shared registry, vendor source or historical receipt was changed.'],
        'operators':operators,'cases':results}
    assert len(operators)==205 and len(results)==2062
    assert len({(r['op'],r['case_id']) for r in results})==len(results)
    assert static['factory_metadata_pass']==sum(r['case_adapter_available'] for r in results)
    assert static['metadata_failures']==0
    (HERE/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    progress={'schema':'tla-progress/v1','owner':'allpairs-remote/tla','phase':'cpu_scope_complete',
        'updated_utc':now,'npu_allowed':False,**summary,
        'coverage_file':'TileOPs/benchmarks/ascend_all_pairs/tla/coverage.json',
        'checks':static,'remaining':['Review deployment path/hash/ELF evidence for this host; runtime and native kernel validation remain separate.',
            'Real NPU numerical validation, full input-preservation checks and native-call ownership receipts require a future explicitly authorized host-locked run.',
            ('Shared factory entries still missing: '+', '.join(missing_factory)) if missing_factory else 'All 205 shared factory names now have AST registration evidence; actual registry import remains untested.',
            'TopKSelect capacity/window adapter remains deployment-blocked; unsupported contracts need different original APIs or separately authorized kernel development.']}
    (HERE/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
