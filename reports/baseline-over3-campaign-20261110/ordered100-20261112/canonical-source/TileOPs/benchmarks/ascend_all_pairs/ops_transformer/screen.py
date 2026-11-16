"""CPU-only case restrictions derived from original API definitions."""
import math

def screen(c):
    op=c['op'];p=c['params'];t=c.get('tensors',p)
    def no(kind,why):return kind,why
    if op in ('AllFwdOp','AnyFwdOp') and t['x'][1]!='bool':
        return no('unsupported_dtype','Exact native min/max maps Boolean data to zero/one losslessly. Arbitrary numeric/complex truth conversion is not a lossless dtype adapter and has no bound own comparison API.')
    if op=='InstanceNormFwdOp' and p.get('use_input_stats',True) and 'running_mean' in t:
        return no('unsupported_output_contract','Original manifest reference returns normalized output plus updated running mean/variance; own NormRopeConcat only exposes normalized tensors. Complete oracle retained; no Partial substitution.')
    if op in ('AmaxFwdOp','AminFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveMaxPool2dFwdOp','MaximumFwdOp','MinimumFwdOp','ReluFwdOp','ClampScalarFwdOp','ClampTensorFwdOp','HardtanhFwdOp') and next(iter(t.values()))[1]=='float32':
        return None  # Original FP32 native selection route; full oracle pending.
    if op in ('AbsFwdOp','MaximumFwdOp','MinimumFwdOp','ReluFwdOp','ClampScalarFwdOp','ClampTensorFwdOp','HardtanhFwdOp','CumsumFwdOp') and next(iter(t.values()))[1] not in ('float16','bfloat16'):
        return no('unsupported_dtype','Original native extrema/prefix route requires FP16/BF16 without narrowing; integer and FP32 templates are not bound.')
    if op in ('KDAFwdOp','GDNFwdOp','DeltaNetFwdOp','DeltaNetRecurrentFwdOp','DeltaNetChunkFwdOp'):
        if t['q'][1]!='bfloat16':return no('unsupported_dtype','Original RecurrentKda/ChunkGatedDeltaRule Q/K/V template is BF16. FP16 conversion would be lossy.')
        if p.get('use_qk_l2norm_in_kernel'):return no('unsupported_math','Original RecurrentKda NormalizeRows divides by sqrt(sum(x*x)) without manifest epsilon=1e-6. Source arch35/recurrent_kda_arch35.h:496; cannot alter native math.')
    if op in ('LayerNormFwdOp','AdaLayerNormFwdOp','AdaLayerNormZeroFwdOp'):
        width=math.prod(p.get('normalized_shape',[t['x'][0][-1]]))
        if width>1024 or width%2:return no('source_adapter_blocked_kernel_limit',f'Original NormRopeConcat headDim must be even and <=1024, manifest complete normalized width is {width}; splitting that axis changes statistics.')
    if op in ('GroupNormFwdOp','InstanceNormFwdOp'):
        shape=t['x'][0];width=math.prod(shape[1:])//p.get('num_groups',shape[1])
        if width>1024 or width%2:return no('source_adapter_blocked_kernel_limit',f'Original NormRopeConcat complete normalization width limit is even <=1024; this case requires {width}.')
        if not p.get('use_input_stats',True):return no('unsupported_math','Original NormRopeConcat cannot select supplied running mean/variance; it always computes input statistics.')
    if op in ('BmmFwdOp','GroupedGemmFwdOp','MoEGroupedGemmFwdOp','Conv1dFwdOp','Conv3dFwdOp','SSDChunkCouplingFwdOp','SiluFwdOp'):
        first=next(iter(t.values()))
        if first[1] not in ('float16','bfloat16'):return no('unsupported_dtype','Chosen original native API has FP16/BF16 input pointers; no lossy FP32/integer input narrowing.')
    if op in ('AmaxFwdOp','AminFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveMaxPool2dFwdOp') and next(iter(t.values()))[1] not in ('float16','bfloat16'):
        return no('unsupported_dtype','Original FlashAttentionScore raw-extrema route takes FP16/BF16 input; no FP32 narrowing.')
    if op=='MulFwdOp' and next(iter(t.values()))[1] not in ('float16','bfloat16','float32'):
        return no('supported_unadapted','Original MhcPost scalar product route only accepts FP16/BF16 h_out pointers. A lossless native FP32 grouped-matrix product mapping is not yet bound.')
    if op=='GeluTanhAndMulFwdOp' and t['x'][1]!='float16':return no('unsupported_dtype','Original BiasGateGelu is hard-coded to half pointers; BF16/FP32 narrowing is forbidden.')
    if (op=='GQAVarlenFwdOp' and p.get('pos_encoding_mode')=='rope' and
        p.get('softcap') is None and not any(k in t for k in ('sinks','q_scale','k_scale','v_scale')) and
        all(t[k][1] in ('float16','bfloat16') for k in ('q','k','v'))):return None
    if op in ('GQAVarlenFwdOp','GQAPagedFwdOp'):
        if p.get('softcap') is not None or p.get('pos_encoding_mode','none')!='none' or any(k in t for k in ('sinks','q_scale','k_scale','v_scale','rope_cos','rope_sin')):
            return no('supported_unadapted','Basic FlashAttentionScore route does not expose this manifest quantization/RoPE/sink/softcap contract. Additional original API mapping is unresolved.')
    if op=='MoEGroupedGemmFwdOp' and p.get('activation')=='silu_and_mul':return None
    if op=='MoEGroupedGemmFwdOp' and p.get('activation') is not None:return no('supported_unadapted','GroupedMatmulAdd lacks the requested fused gated activation; original fused MLP route is not deployed for 950.')
    if op=='MoEPostPermuteFwdOp' and (p.get('epilogue') is not None or p.get('out_dtype') not in (None,t['expert_output'][1])):return no('supported_unadapted','Original MoeFinalizeRoutingV2 adapter has not mapped the manifest extra routed epilogue or output dtype.')
    if op=='PagedKVCacheWriteFwdOp' and any(k in t for k in ('k_scale','v_scale')):return no('unsupported_math','Original ScatterPaKvCache has no independent per-cache FP8 scaling/saturation input.')
    if op=='PagedKVCacheGatherFwdOp' and ('scale' in t or t['dst'][1]!=t['cache'][1]):return no('unsupported_math','Original GatherPaKvCache copies cache bits and does not expose scaled FP8 dequantization.')
    if op in ('GemmFP8FwdOp','BmmFP8FwdOp','GemmINT8W8A8FwdOp'):
        m=t['a'][0][-2];n=t['b'][0][-1] if op=='BmmFP8FwdOp' and not p['trans_b'] else t['b'][0][-2]
        if op=='GemmFP8FwdOp' and t['scale_a'][0]==[m,-(-t['a'][0][-1]//128)] and t['scale_b'][0]==[-(-n//128),-(-t['a'][0][-1]//128)]:return None
        if math.prod(t['scale_a'][0]) not in (1,m) or math.prod(t['scale_b'][0]) not in (1,n):return no('supported_unadapted','Original G-B quantization exists, but this adapter only binds exact T-T/K-C scale layouts; blockwise native metadata packing is unresolved.')
    if op=='GQAPrefillPagedWithKVCacheFwdOp' and p.get('fuse_rope') and not p.get('softcap') and not any('float8' in v[1] for v in t.values()):return None
    if op=='GQAPrefillPagedWithKVCacheFwdOp' and (p.get('softcap') or p.get('fuse_rope') or any('float8' in v[1] for v in t.values())):return no('supported_unadapted','Basic original append/attention route does not yet map fused RoPE, softcap or scaled FP8 cache.')
    if op=='MLAKVCacheWriteFwdOp' and t['kv_cache'][1]!=t['kv_c'][1]:return no('supported_unadapted','Original cache scatter lacks scaled FP8 conversion; dtype narrowing is not a valid substitute.')
    if op=='MLAPagedFwdOp' and t['q'][1]!=t['kv_cache'][1]:return no('supported_unadapted','Original FlashMlaWithKvcache lacks independent FP8 cache scaling; only nonquantized original MLA route is bound.')
    return None
