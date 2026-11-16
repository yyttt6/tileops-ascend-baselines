"""CPU-only, reproducible per-op/case source audit and incremental receipt index."""
import ast,collections,datetime,hashlib,json,inspect,sys
from screen import screen
from pathlib import Path
ROOT=Path('/root/dyq/workspace')
OUT=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer'
SRC=ROOT/'ascend_baselines/ops-transformer'
PKG=ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer'
sys.path.insert(0,str(ROOT/'TileOPs'))
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
INV=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/inventory.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def evidence(relative):
    p=SRC/relative
    if p.is_dir():
        candidates=list(p.glob('README.md'))+list(p.glob('Readme.md'))+list(p.glob('op_host/*def.cpp'))+list(p.glob('op_api/*.h'))+list(p.glob('op_host/op_api/*.h'))
        if not candidates:candidates=[q for q in p.rglob('*.cpp')][:1]
    else:candidates=[p]
    records=[]
    for q in candidates:
        if not q.is_file():continue
        lines=q.read_text(errors='replace').splitlines()
        line=next((i for i,t in enumerate(lines,1) if 'OP_ADD(' in t or 'TORCH_LIBRARY_IMPL(' in t),1)
        records.append({'file':str(q.relative_to(ROOT)),'line':line,'sha256':sha(q)})
    return records

catalog=[]
for section in ('attention','posembedding','gmm','moe','mamba','mhc','ffn','mc2','experimental'):
    root=SRC/section
    for p in sorted(root.rglob('*def.cpp')):
        if any(s in p.parts for s in ('3rd','third_party','tests','build')):continue
        text=p.read_text(errors='replace')
        import re
        catalog.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p),'registered_ops':re.findall(r'OP_ADD\((\w+)\)',text),'attributes':re.findall(r'Attr\("([^"]+)"\)',text)})
cat_path=OUT/'source_catalog.json'
cat_path.write_text(json.dumps({'schema':'ops-transformer-own-source-catalog/v1','scope':'Own registered op_def.cpp; excludes 3rd, stubs, external wheel implementations','entries':catalog},indent=2)+'\n')
source_map={
 'BmmFwdOp':'gmm/grouped_matmul_add','GemmFwdOp':'gmm/grouped_matmul_add','Conv1dFwdOp':'gmm/grouped_matmul_add','Conv2dFwdOp':'gmm/grouped_matmul_add','Conv3dFwdOp':'gmm/grouped_matmul_add',
 'GroupedGemmFwdOp':'gmm/grouped_matmul_add','MoEGroupedGemmFwdOp':'gmm/grouped_matmul_add','SumFwdOp':'gmm/grouped_matmul_add','MeanFwdOp':'gmm/grouped_matmul_add','SSDChunkCouplingFwdOp':'gmm/grouped_matmul_add',
 'AddFwdOp':'experimental/moe/layernorm_add_add_quant','RMSNormFwdOp':'posembedding/norm_rope_concat','LayerNormFwdOp':'posembedding/norm_rope_concat','AmaxFwdOp':'attention/flash_attention_score',
 'GQADenseFwdOp':'attention/flash_attention_score','GQAVarlenFwdOp':'attention/flash_attention_score','GQAPagedFwdOp':'attention/flash_attention_score','GQABwdOp':'attention/flash_attention_score_grad',
 'MoEPrePermuteFwdOp':'moe/moe_init_routing_v2','MoEPostPermuteFwdOp':'moe/moe_finalize_routing_v2','FusedTopKFwdOp':'moe/moe_fused_topk','SoftmaxFwdOp':'moe/moe_gating_top_k_softmax',
 'PagedKVCacheWriteFwdOp':'attention/scatter_pa_kv_cache','PagedKVCacheGatherFwdOp':'attention/gather_pa_kv_cache','MergeAttentionStatesFwdOp':'attention/attention_update',
 'MHCPostFwdOp':'mhc/mhc_post','MHCPreFwdOp':'mhc/mhc_pre','RoPEFwdOp':'posembedding/rotary_position_embedding','RoPELlama31FwdOp':'posembedding/rotary_position_embedding','YaRNFwdOp':'posembedding/rotary_position_embedding','LongRoPEFwdOp':'posembedding/rotary_position_embedding','RoPENeoxPositionIdsFwdOp':'posembedding/rotary_position_embedding',
 'DeltaNetRecurrentFwdOp':'attention/recurrent_gated_delta_rule','DeltaNetFwdOp':'attention/chunk_kda_fwd','DeltaNetChunkFwdOp':'attention/chunk_kda_fwd','KDAFwdOp':'attention/chunk_kda_fwd','GDNFwdOp':'attention/chunk_gated_delta_rule',
 'CausalConv1dFwdOp':'mamba/causal_conv1d','CausalConv1dUpdateFwdOp':'mamba/causal_conv1d','SSDChunkStateFwdOp':'experimental/mamba/mamba2_chunk_state','SSDChunkScanFwdOp':'experimental/mamba/mamba2_chunk_scan','SSDChunkCumsumFwdOp':'experimental/mamba/mamba2_chunk_cumsum','SSDStatePassingFwdOp':'experimental/mamba/mamba2_chunk_state_passing',
 'FP8QuantFwdOp':'attention/indexer_quant_cache','FP8QuantPerBlockFwdOp':'attention/indexer_quant_cache','GemmFP8FwdOp':'gmm/grouped_matmul','BmmFP8FwdOp':'gmm/grouped_matmul','GemmINT8W8A8FwdOp':'gmm/grouped_matmul','GemmW4A16FwdOp':'gmm/grouped_matmul','GemmW4A8FwdOp':'gmm/grouped_matmul','MoEGroupedGemmFP8FwdOp':'gmm/grouped_matmul',
 'DSADecodeWithKVCacheFwdOp':'attention/sparse_flash_attention','DSAPagedFwdOp':'attention/sparse_flash_attention','MLAPagedFwdOp':'attention/flash_mla_with_kvcache','MLADecodeWithKVCacheFwdOp':'attention/flash_mla_with_kvcache','MLAVarlenFwdOp':'attention/flash_attention_score',
 'FP8LightningIndexerFwdOp':'attention/quant_lightning_indexer','NSAVarlenFwdOp':'attention/nsa_selected_attention','NSACompressedVarlenFwdOp':'attention/nsa_compress_attention','NSATopKVarlenFwdOp':'attention/lightning_indexer','TopKSelectFwdOp':'attention/lightning_indexer',
 'GQAPrefillPagedWithKVCacheFwdOp':'attention/fused_infer_attention_score','MHADecodePagedWithKVCacheFwdOp':'attention/incre_flash_attention','MLAKVCacheWriteFwdOp':'attention/scatter_pa_kv_cache',
 'MoEExpertMLPFwdOp':'ffn/ffn','IndexedExpertMLPFwdOp':'ffn/ffn','SharedExpertMLPFwdOp':'ffn/ffn','FusedMoEExpertsFwdOp':'ffn/ffn','FusedMoEFwdOp':'ffn/ffn','FusedMoESharedExpertFwdOp':'ffn/ffn',
 'EngramGateConvFwdOp':'experimental/engram/engram_gate_dsl','EngramGateConvBwdOp':'experimental/engram/engram_gate_dsl','EngramDecodeFwdOp':'experimental/engram/engram_gate_dsl',
 'GeluFwdOp':'experimental/moe/biasgategelu','GeluAndMulFwdOp':'experimental/moe/biasgategelu','GeluTanhAndMulFwdOp':'experimental/moe/biasgategelu','SiluFwdOp':'ffn/ffn','SiluAndMulFwdOp':'ffn/ffn',
 'FusedAddLayerNormFwdOp':'experimental/moe/layernorm_add_add_quant','FusedAddRMSNormFwdOp':'experimental/mamba/mamba2_rmsnormgated','FusedAddRMSNormPerTokenQuantFwdOp':'posembedding/qkv_rms_norm_rope_cache','FusedQKNormRoPEFwdOp':'posembedding/qkv_rms_norm_rope_cache',
 'GroupNormFwdOp':'posembedding/norm_rope_concat','InstanceNormFwdOp':'posembedding/norm_rope_concat','AdaLayerNormFwdOp':'posembedding/norm_rope_concat','AdaLayerNormZeroFwdOp':'posembedding/norm_rope_concat',
}
explicit={
 'MHCPreFwdOp':('unsupported_math','Original MhcPre invRms=(mean(x^2)+eps)^-1/2 differs from manifest sqrt(sum(x*x))/sqrt(nD)+eps outside sqrt; manifest also returns pre-mixed residual and performs distinct sinkhorn contract.'),
 'EngramGateConvFwdOp':('unsupported_ownership','Inspected engram wrapper delegates numerical implementation to external cannbot-arena-net-ops wheel; also lacks manifest convolution output. Cannot attribute dependency implementation to this library.'),
 'EngramGateConvBwdOp':('unsupported_ownership','No own convolution backward implementation; engram wrapper imports external wheel.'),
 'EngramDecodeFwdOp':('unsupported_ownership','External-wheel engram gate wrapper does not supply own projected decode and stateful convolution.'),
 'SSDChunkStateFwdOp':('numerical_failure','Original native SSD kernel remains quarantined by full-output and changed-input checks; no kernel-math repair authorized.'),
 'SSDChunkCumsumFwdOp':('unsupported_dtype','Inspected original chunk_cumsum consumes half dt/dt_bias and fixed softplus path; manifest dt is FP32. Narrowing input changes data and is forbidden.'),
 'FusedAddLayerNormFwdOp':('unsupported_math','Original LayernormAddAddQuant exposes quantized activation and raw residual, not the required floating normalized output; quantization cannot be undone losslessly.'),
 'FusedAddRMSNormPerTokenQuantFwdOp':('unsupported_math','QKV RMS/RoPE/cache operators do not expose manifest residual_out plus row scale and quantized complete output contract.'),
}
explicit.update({
 'MergeAttentionStatesFwdOp':('unsupported_math','Own AttentionUpdate applies natural Exp/Ln (op_kernel/decode_update.h); manifest consumes/returns base-2 LSE. Native API has no logarithm-base attribute; converting input LSE with torch arithmetic is outside allowed layout-only adaptation.'),
 'GeluFwdOp':('unsupported_math','Manifest uses exact erf GELU; own original BiasGateGelu invokes AscendC tanh GELU approximation and provides no approximation selector. No kernel math edits authorized.'),
 'GeluAndMulFwdOp':('unsupported_math','Manifest fused gate uses exact erf GELU; own BiasGateGelu uses tanh approximation without a mode selector. Distinct GeluTanhAndMul route is implemented separately.'),
})
for _op in ('MoEExpertMLPFwdOp','IndexedExpertMLPFwdOp','SharedExpertMLPFwdOp','FusedMoEExpertsFwdOp','FusedMoEFwdOp','FusedMoESharedExpertFwdOp'):
    explicit[_op]=('deployment_blocked_platform','Own FFN README explicitly excludes Ascend950 and original op_def has no 950 configuration. Target-specific deployment remains unresolved; this is not a mathematical unsupported claim. Wrapper-only SwiGLU compositions backed by external CANN operators do not supply own-library numerical ownership.')
implemented={'BmmFwdOp','LayerNormFwdOp','Conv1dFwdOp','Conv3dFwdOp','GQAVarlenFwdOp','GQAPagedFwdOp','MHCPostFwdOp','SoftmaxFwdOp','MoEPostPermuteFwdOp','PagedKVCacheWriteFwdOp','DeltaNetRecurrentFwdOp','FusedTopKFwdOp','SumFwdOp','MeanFwdOp','GroupedGemmFwdOp','MoEGroupedGemmFwdOp','RoPELlama31FwdOp','YaRNFwdOp','LongRoPEFwdOp','RoPENeoxPositionIdsFwdOp','SSDChunkCouplingFwdOp'}
implemented.update({'GQABwdOp','DeltaNetChunkFwdOp','DeltaNetFwdOp','GDNFwdOp','PagedKVCacheGatherFwdOp'})
implemented.update({'CausalConv1dFwdOp','CausalConv1dUpdateFwdOp','KDAFwdOp','GroupNormFwdOp','InstanceNormFwdOp','GeluTanhAndMulFwdOp','SiluFwdOp','GemmFP8FwdOp','BmmFP8FwdOp','GemmINT8W8A8FwdOp'})
source_map.update({op:'attention/recurrent_kda' for op in ('KDAFwdOp','GDNFwdOp','DeltaNetFwdOp','DeltaNetRecurrentFwdOp')})
source_map['SiluFwdOp']='mamba/causal_conv1d'
implemented.update({'MLADecodeWithKVCacheFwdOp','MHADecodePagedWithKVCacheFwdOp'})
source_map.update({op:'attention/flash_attention_score' for op in ('MLADecodeWithKVCacheFwdOp','MHADecodePagedWithKVCacheFwdOp')})
implemented.add('DSADecodeWithKVCacheFwdOp')
source_map['DSADecodeWithKVCacheFwdOp']='attention/flash_attention_score'
implemented.add('SSDStatePassingFwdOp')
implemented.update({'GQAPrefillPagedWithKVCacheFwdOp','NSAVarlenFwdOp'})
source_map.update({op:'attention/flash_attention_score' for op in ('GQAPrefillPagedWithKVCacheFwdOp','NSAVarlenFwdOp')})
implemented.add('NSACompressedVarlenFwdOp')
source_map['NSACompressedVarlenFwdOp']='attention/flash_attn'
implemented.update({'MLAKVCacheWriteFwdOp','MLAVarlenFwdOp'})
source_map['MLAVarlenFwdOp']='attention/flash_attn'
implemented.add('FusedQKNormRoPEFwdOp')
source_map['FusedQKNormRoPEFwdOp']='posembedding/norm_rope_concat'
implemented.add('TopKSelectFwdOp')
source_map['TopKSelectFwdOp']='moe/moe_gating_top_k_softmax'
implemented.add('MoEGroupedGemmFP8FwdOp')
explicit.pop('MergeAttentionStatesFwdOp',None)
implemented.add('MergeAttentionStatesFwdOp')
implemented.add('SSDChunkScanFwdOp')
explicit['SSDChunkCumsumFwdOp']=('unsupported_math','Original native frontend internally casts dt/bias to FP16 and kernel hard-clamps softplus at CLAMP_MAX=10000000 (CustVec.h:26,249), whereas all manifest cases specify dt_max=inf. Native original source was successfully retargeted/compiled without math edits, but there is no unclamped API mode.')
implemented.add('GemmW4A16FwdOp')
implemented.add('HadamardTransformFwdOp')
source_map['HadamardTransformFwdOp']='gmm/grouped_matmul'
source_map['NSATopKVarlenFwdOp']='attention/nsa_compress_attention'
explicit['GemmW4A8FwdOp']=('missing_reference_contract','Parent-added shared factory explicitly returns ReferenceInfeasible: packed INT4 byte order and BF16 group-scale/intermediate rounding contract are undefined. No W4A16 or vLLM packing convention may be invented as its oracle. See workloads/ascend_extended_transform_quant.py W4A8_REFERENCE_BLOCKER.')
explicit.update({
 'DSAPagedFwdOp':('supported_unadapted','Original SparseFlashAttention/FlashMla APIs take typed half/BF16 caches; manifest requires the V3.2 raw uint8 656-byte row (four FP32 per-token/group scales, FP8 latent and BF16 PE), sparse selection and complete LSE. Own-library codec/dequantization plus LSE binding remains unresolved; no external FP8 decode fallback is used.'),
 'FP8LightningIndexerFwdOp':('unsupported_output_contract','Original QuantLightningIndexer exposes only INT32 sparse_indices (op_def:126); LightningIndexer sparse_values are BF16/FP16 (op_def:61), not the required full FP32 weighted ReLU score tensor. Original quantized and nonquantized APIs cannot losslessly recover the complete manifest output by layout/crop.'),
 'FP8QuantPerBlockFwdOp':('unsupported_math','Original IndexerQuantCache dynamic FP8 kernel computes scale from rowMax with no all-zero scale=1 rule (arch35/indexer_quant_cache_base.h:409-427). Flattening each 128x128 block is legal, but complete scale output differs for zero groups and the original input controls include zeros. No arithmetic correction or kernel edits are permitted.'),
 'FusedAddRMSNormFwdOp':('source_adapter_blocked_kernel_limit','Original NormRopeConcat complete RMS width is even <=1024; all manifest widths are 4096..16384. Original Mamba2Rmsnormgated additionally applies mandatory SiLU(z) and cannot select a unit gate/no-gate mode. Splitting normalization axes or external normalized arithmetic is not a layout adapter.'),
 'NSATopKVarlenFwdOp':('supported_unadapted','Own NsaCompressAttention exposes top-k but has only 910B/910_93 deployment configurations; its sliding-block importance/topk-mask API needs a verified mapping to the manifest closed-block denominator and first/current/previous priority rules. Own 950 LightningIndexer uses weighted ReLU scores, not this softmax importance. No unsupported-by-example-shape claim.'),
 'SiluAndMulFwdOp':('supported_unadapted','Own CausalConv1d supplies native SiLU, but returns input dtype before a separate multiplication; complete FP32 fused gate rounding and original native multiplication route are unresolved. Own FFN supports SwiGLU but has no Ascend950 configuration. No Torch numerical fallback or tolerance change.'),
})
implemented.update({'MLAPagedFwdOp','Conv2dFwdOp','SigmoidFwdOp'})
source_map['SigmoidFwdOp']='moe/moe_fused_topk'
implemented.add('DSAPagedFwdOp')
explicit.pop('DSAPagedFwdOp',None)
source_map['DSAPagedFwdOp']='attention/flash_mla_with_kvcache'
implemented.add('MulFwdOp')
source_map['MulFwdOp']='mhc/mhc_post'
implemented.update({'NegFwdOp','SubFwdOp','LogSumExpFwdOp','LogSoftmaxFwdOp'})
source_map.update({op:'attention/attention_update' for op in ('LogSumExpFwdOp','LogSoftmaxFwdOp')})
source_map.update({op:'gmm/grouped_matmul' for op in ('NegFwdOp','SubFwdOp')})
implemented.update(('AvgPool1dFwdOp', 'AvgPool2dFwdOp', 'AvgPool3dFwdOp', 'AdaptiveAvgPool2dFwdOp', 'MeanPoolingFwdOp'))
source_map.update({op:'gmm/grouped_matmul' for op in ('AvgPool1dFwdOp', 'AvgPool2dFwdOp', 'AvgPool3dFwdOp', 'AdaptiveAvgPool2dFwdOp', 'MeanPoolingFwdOp')})
implemented.update({'AmaxFwdOp','AminFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveMaxPool2dFwdOp'})
source_map.update({op:'attention/flash_attention_score' for op in ('AmaxFwdOp','AminFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveMaxPool2dFwdOp')})
implemented.discard('DeltaNetChunkFwdOp')
implemented.update({'AbsFwdOp','MaximumFwdOp','MinimumFwdOp','ReluFwdOp','ClampScalarFwdOp','ClampTensorFwdOp','HardtanhFwdOp','CumsumFwdOp'})
source_map.update({op:'attention/flash_attention_score' for op in ('AbsFwdOp','MaximumFwdOp','MinimumFwdOp','ReluFwdOp','ClampScalarFwdOp','ClampTensorFwdOp','HardtanhFwdOp')})
source_map['CumsumFwdOp']='gmm/grouped_matmul_add'
implemented.add('LerpScalarFwdOp')
source_map['LerpScalarFwdOp']='gmm/grouped_matmul_add'
implemented.update({'ArgmaxFwdOp','ArgminFwdOp'})
source_map.update({op:'moe/moe_gating_top_k_softmax_v2' for op in ('ArgmaxFwdOp','ArgminFwdOp','MaxPool1dIndicesFwdOp','MaxPool2dIndicesFwdOp','MaxPool3dIndicesFwdOp','AdaptiveMaxPool2dIndicesFwdOp')})
for _op in ('MaxPool1dIndicesFwdOp','MaxPool2dIndicesFwdOp','MaxPool3dIndicesFwdOp','AdaptiveMaxPool2dIndicesFwdOp'):
    explicit[_op]=('supported_unadapted','Native top-1 exposes indices and the raw-max pool layout is implemented. Full first-occurrence tie ordering and conversion of window indices to original spatial indices need a verified binding; no unsupported-by-example-shape claim.')
for _op in ('MaxPool1dIndicesFwdOp','MaxPool2dIndicesFwdOp','MaxPool3dIndicesFwdOp','AdaptiveMaxPool2dIndicesFwdOp'):
    implemented.add(_op)
    explicit.pop(_op,None)
explicit['DeltaNetChunkFwdOp']=('unsupported_output_contract','Unchanged manifest requires six chunk outputs including WY intermediates; own ChunkGatedDeltaRule only returns out/final_state. Original full-output checker rejected the incomplete adapter; no Partial oracle substitute. See delta_chunk_output_contract.json.')
implemented.update({'AddFwdOp','ExpFwdOp','Expm1FwdOp','AllFwdOp','AnyFwdOp'})
source_map.update({op:'moe/moe_gating_top_k_softmax_v2' for op in ('AllFwdOp','AnyFwdOp')})
source_map.update({op:'experimental/mamba/mamba2_chunk_state_passing' for op in ('ExpFwdOp','Expm1FwdOp')})
implemented.update(('HardsigmoidFwdOp', 'HardswishFwdOp', 'LeakyReluFwdOp', 'PreluFwdOp', 'LerpTensorFwdOp'))
source_map.update({op:'moe/moe_finalize_routing_v2' for op in ('HardsigmoidFwdOp', 'HardswishFwdOp', 'LeakyReluFwdOp', 'PreluFwdOp', 'LerpTensorFwdOp')})
implemented.add('SiluAndMulFwdOp')
explicit.pop('SiluAndMulFwdOp',None)
source_map['SiluAndMulFwdOp']='moe/moe_gating_top_k'
old=json.loads((ROOT/'TileOPs/benchmarks/ascend/classic_matrix.json').read_text())
oldrows={(r['op'],r['case_id']):r for r in old['rows'] if r['library']=='ops-transformer'}
receipts={}
for p in sorted(OUT.rglob('*.json')):
    if p.name in ('coverage.json','progress.json','source_catalog.json'):continue
    try:r=json.loads(p.read_text())
    except (ValueError,OSError):continue
    if isinstance(r,dict) and r.get('op') and r.get('case_id') and r.get('status'):
        receipts.setdefault((r['op'],r['case_id']),[]).append((p,r))
x=json.loads(INV.read_text());entries=[];missing_factories=set(x.get('missing_factory_ops',[]))
# Parent has added these exact shared factories after the inventory snapshot.
extended=ROOT/'TileOPs/benchmarks/_cases/ascend_extended.py'
if extended.exists():
    for node in ast.walk(ast.parse(extended.read_text())):
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith('Op'):missing_factories.discard(node.value)
for entry in x['operators']:
    op=entry['op'];cases=[c for c in x['cases'] if c['op']==op]
    src=source_map.get(op)
    if op in implemented:status,reason='supported_untested','Layout/API adapter implemented; acceptance requires full manifest and input protection receipts.'
    elif op in source_map:status,reason='source_match_unimplemented','Related original API identified; full manifest contract mapping and deployment/device acceptance are not yet complete. This is not a claim of mathematical impossibility.'
    else:status,reason='unsupported_within_scope','No legal complete-output binding for '+op+' was identified in the inspected own exposed APIs/compositions. Internal primitives and common/stub/3rd dependency operators are excluded; this is a scoped source-audit conclusion, not a mathematical impossibility proof.'
    if op in explicit:status,reason=explicit[op]
    ev=evidence(src) if src else evidence('README.md')
    if op in ('AmaxFwdOp','AminFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveMaxPool2dFwdOp','MaximumFwdOp','MinimumFwdOp','ReluFwdOp','ClampScalarFwdOp','ClampTensorFwdOp','HardtanhFwdOp'):
        ev += evidence('moe/moe_gating_top_k_softmax_v2')
    if op=='SoftmaxFwdOp':ev += evidence('moe/moe_gating_top_k_softmax_v2')
    if op=='SiluAndMulFwdOp':ev += evidence('moe/moe_finalize_routing_v2')
    if op=='LogSoftmaxFwdOp':ev += evidence('moe/moe_finalize_routing_v2')
    if op in ('HardsigmoidFwdOp', 'HardswishFwdOp', 'LeakyReluFwdOp', 'PreluFwdOp'):ev += evidence('moe/moe_gating_top_k_softmax_v2')
    if op in ('SigmoidFwdOp','FusedTopKFwdOp'):ev += evidence('moe/moe_gating_top_k')
    if op in ('AddFwdOp','MulFwdOp','NegFwdOp','SubFwdOp','LerpScalarFwdOp','SumFwdOp','MeanFwdOp'):ev += evidence('moe/moe_finalize_routing_v2')
    ev.append({'file':str(cat_path.relative_to(ROOT)),'line':1,'sha256':sha(cat_path)})
    cr=[]
    for c in cases:
        row={'case_id':c['case_id'],'status':status,'reason':reason,'device_executed':False,'receipt':None}
        restriction=screen(c)
        if restriction:row.update(status=restriction[0],reason=restriction[1])
        key=(op,c['case_id']);hist=oldrows.get(key)
        if hist:
            row['historical_classic_record']=hist
            if hist['status']=='verified':row.update(status='verified_historical',reason=hist['reason'],device_executed=True,receipt=hist.get('last_receipt'),ownership_review=hist.get('root_review'))
            elif op not in implemented:row.update(status=hist['status'],reason=hist['reason'],device_executed=hist.get('device_executed',False),receipt=hist.get('last_receipt'))
        if key in receipts:
            records=receipts[key];p,r=max(records,key=lambda pr:pr[1].get('finished_at',''))
            rs=r['status'];st=rs
            ownership_path=p.with_suffix('.ownership.json')
            ownership=json.loads(ownership_path.read_text()) if ownership_path.exists() else {}
            if rs=='verified' and not ownership.get('accepted'):st='numerically_verified_ownership_pending'
            if rs=='unsupported':st=(screen(c) or ('supported_unadapted',''))[0]
            if rs=='failed':st='numerical_failure' if r.get('error_type')=='AssertionError' else ('deployment_blocked' if r.get('stage')=='build' else 'device_failed')
            if rs=='failed' and r.get('error_type')!='AssertionError':
                error=r.get('error','')
                if 'returned 500002' in error:st='deployment_or_binding_failed'
                elif 'unexpected keyword argument' in error or 'Is a directory' in error:st='adapter_failed'
                elif 'reference(*move' in r.get('traceback',''):st='reference_transport_failed'
            calls=r.get('native_graph_calls',[])+r.get('native_dispatch_calls',[])
            row.update(status=st,reason=r.get('error',r.get('reason','Original complete manifest and input protection checks passed.')),device_executed=bool(calls),receipt=str(p.relative_to(ROOT)),api=r.get('api'),source_sha256=r.get('source_sha256',{}),verification=r.get('verification'),native_call_count=len(calls))
            row['ownership_receipt']=str(ownership_path.relative_to(ROOT)) if ownership_path.exists() else None
            row['all_receipts']=[str(v[0].relative_to(ROOT)) for v in records]
            row['executed_adapter_snapshots']=r.get('source_snapshots',{})
            row['adapter_files_changed_since_execution']=[name for name,digest in r.get('source_sha256',{}).items() if name.startswith('TileOPs/benchmarks/ascend_all_pairs/ops_transformer/') and (not (ROOT/name).is_file() or sha(ROOT/name)!=digest)]
        if op=='InstanceNormFwdOp' and c['params'].get('use_input_stats',True) and 'running_mean' in c.get('tensors',c['params']):
            kind,why=screen(c)
            row.update(status=kind,reason=why)
        if op=='DeltaNetChunkFwdOp':
            row.update(status=explicit[op][0],reason=explicit[op][1])
        row['source_evidence']=ev
        row['registered']=op in PROVIDERS
        row['adapter']=str(Path(inspect.getsourcefile(PROVIDERS[op])).relative_to(ROOT)) if op in PROVIDERS else None
        cr.append(row)
    if op in missing_factories or not cases:
        status='missing_workload_factory';reason='Inventory records no shared workload factory. Raw manifest cases exist but cannot be tested without an original factory.'
        for c in cr:c.update(status=status,reason=reason)

    elif cr:
        statuses=collections.Counter(r['status'] for r in cr)
        if len(statuses)==1:status=next(iter(statuses))
        elif any(s.startswith('verified') for s in statuses):status='partially_verified'
        else:status='mixed_case_status'
    adapter_path=str(Path(inspect.getsourcefile(PROVIDERS[op])).relative_to(ROOT)) if op in PROVIDERS else None
    entries.append({'op':op,'category':entry['category'],'status':status,'reason':reason,'candidate_source':src,'source_evidence':ev,'adapter':adapter_path,'registered':op in PROVIDERS,'cases':cr,'case_status_counts':dict(collections.Counter(c['status'] for c in cr))})
summary={'operators':len(entries),'cases':sum(len(r['cases']) for r in entries),'op_status_counts':dict(collections.Counter(r['status'] for r in entries)),'case_status_counts':dict(collections.Counter(c['status'] for r in entries for c in r['cases']))}
result={'schema':'ops-transformer-all-pairs-coverage/v1','schema_notes':{'verified':'New full manifest receipt, input preservation and accepted own-source compilation/launch attribution for its recorded source hashes/snapshots. Later source revisions do not inherit that numerical evidence; adapter_files_changed_since_execution exposes byte changes, which may include unrelated functions in shared modules.','numerically_verified_ownership_pending':'Full numerical checks passed but native ownership acceptance is not yet complete','verified_historical':'Read-only classic verified receipt and owner review retained; not newly executed','source_match_unimplemented':'Related own API, unresolved mapping; not unsupported and not registered','supported_untested':'Adapter exists but no complete acceptance receipt','unsupported_math':'Explicit contract mismatch','unsupported_within_scope':'No equivalent own exposed API in inspected catalog','missing_workload_factory':'Inventory case count zero; testing cannot be fabricated'},'library':'ops-transformer','updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_inventory_sha256':sha(INV),'summary':summary,'operators':entries}
(OUT/'coverage.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
progress=json.loads((OUT/'progress.json').read_text());progress.update(updated_at=result['updated_at'],coverage_summary=summary,coverage_file=str((OUT/'coverage.json').relative_to(ROOT)),implemented_new_ops=sorted(implemented),registered_ops=sorted(PROVIDERS),new_verified=summary['case_status_counts'].get('verified',0));(OUT/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
print(json.dumps(summary))
