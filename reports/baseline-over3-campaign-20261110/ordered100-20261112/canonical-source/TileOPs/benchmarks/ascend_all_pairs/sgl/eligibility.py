"""CPU-only source restrictions used to plan cases, never to claim validation."""

def restriction(case):
    op=case['op'];p=case['params'];ts=case['tensors']
    def dtype(*names):
        for name in names:
            if name in ts:return ts[name][1]
        return None
    if op in ('ReciprocalFwdOp','DivFwdOp'):
        if p.get('rounding_mode') is not None:
            return 'Rounded division needs an original native floor/truncation stage that this route does not expose.'
        if any(v[1] not in ('float16','bfloat16','float32') for k,v in ts.items() if k!='output'):
            return 'Singleton RMS reciprocal is scoped to the unchanged positive floating manifest domain, not integer division.'
    if op=='StdFwdOp' or op=='VectorNormFwdOp' and p.get('ord')==2:
        import math
        shape,kind=ts['x'];dims=p.get('dim')
        dims=list(range(len(shape))) if dims is None else ([dims] if isinstance(dims,int) else dims)
        width=math.prod(shape[d] for d in dims)
        if width*(4 if kind=='float32' else 2)>65536:
            return 'Original complete LayerNorm public API rejects a feature row larger than 64KiB.'
    if op in ('LogicalAndFwdOp','LogicalOrFwdOp','LogicalNotFwdOp') and any(v[1] not in ('bool','float16','bfloat16','float32') for k,v in ts.items() if k!='output'):
        return 'Native finite zero-comparison handles floating and boolean truth; complex or integer operands would require unimplemented lossless predicate semantics.'
    if op in ('BitwiseAndFwdOp','BitwiseOrFwdOp','BitwiseXorFwdOp') and any(v[1]!='bool' for v in ts.values()):
        return 'Only exact boolean 0/1 native arithmetic composition is registered; numeric truth predicates and integer bit arithmetic are not supplied by this route.'
    if op=='VectorNormFwdOp' and p.get('ord') not in (1,2,float('inf')):
        return 'Complete native compositions currently cover L1, L2 and infinity norms only.'
    if op=='GemmFP8FwdOp' and any(ts[name][0]!=[1,1] for name in ('scale_a','scale_b')):
        return 'K-dependent FP8 block scales cannot move outside complete native SGEMMV; narrowing scaled FP32 operands is not lossless.'
    if op in ('BmmFwdOp','GroupedGemmFwdOp','Conv1dFwdOp','Conv3dFwdOp') and dtype('a','input','x') not in ('float16','bfloat16'):
        return 'Own SGL vector LoRA only accepts matching FP16/BF16 operands; conversion from FP32/FP8 is not lossless.'
    if op=='KDAFwdOp' and dtype('q')!='bfloat16':
        return 'Own chunk KDA native input contract requires BF16; existing FP16 target_verify has a mandatory different Q/K normalization epsilon, so no unchanged equivalent route is registered.'
    if op=='GDNFwdOp' and dtype('q')=='float16' and any(p.get(k) for k in ('use_gate_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval')):
        return 'Original FP16 full GDN pipeline accepts activated gates/beta; the raw-gate alternate API has different Q/K normalization semantics.'
    if op in ('GQADenseFwdOp','GQAVarlenFwdOp','GQAPagedFwdOp','GQAPrefillPagedWithKVCacheFwdOp'):
        if p.get('softcap') not in (None,0):return 'Original own decode GQA/sink API has no logits softcap parameter.'
        if any(v[1].startswith('float8') for v in ts.values()):return 'Unquantized own decode/store route cannot express supplied FP8 scales; existing full-FP8 classic path remains numerically quarantined.'
    if op=='PagedKVCacheWriteFwdOp' and (any(v[1].startswith('float8') for v in ts.values()) or 'k_scale' in ts or 'v_scale' in ts):
        return 'Original SGL slot store copies values and has no cache quantization divisor/scale semantics.'
    if op=='MLAKVCacheWriteFwdOp' and ('scale' in ts or dtype('kv_c')!=dtype('kv_cache')):
        return 'Original SGL slot store has no FP8 cache conversion or scale parameter.'
    if op=='BatchNormFwdOp':
        import math
        shape,kind=ts['x']
        if not p['training']:return 'Original grouped normalization requires input statistics, not supplied running statistics.'
        if math.prod(shape)//shape[1]*(4 if kind=='float32' else 2)>65536:return 'Original unchanged public LayerNorm host rejects complete feature rows larger than64KiB.'
    if op=='InstanceNormFwdOp' and not p.get('use_input_stats',True):
        return 'Original grouped LayerNorm always computes input statistics; running-statistics inference needs another complete API.'
    if op in ('MoEGroupedGemmFwdOp','MoEExpertMLPFwdOp') and p.get('activation') not in (None,'silu_and_mul'):
        return 'Own expert composition has native SwiGLU but no native GELU epilogue.'
    return None
