"""Proven source-contract exclusions, kept separate from deployment failures."""
from math import prod

def reason(op,p):
    if op=='InstanceNormFwdOp' and (not p.get('use_input_stats',True) or p.get('running_mean') is not None or p.get('running_var') is not None):
        return 'Original LayerNorm neither updates nor returns the declared InstanceNorm running statistics and cannot use stored inference statistics; the complete tracked/eval output contract is unsupported.'
    if op=='MoEGroupedGemmFwdOp' and p.get('activation') is not None:
        return 'Original weighted reduction has no fused gated activation epilogue.'
    if op in ('AllFwdOp','AnyFwdOp') and p['x'][1] not in ('bool','float16','bfloat16','float32'):
        return 'Original ReduceAll/Any graph lacks this integer/complex dtype; there is no audited lossless supported widening for the full domain.'
    if op in ('VectorNormFwdOp','CountNonzeroFwdOp'):
        shape,dtype=p['x']
        if dtype not in ('bool','float16','bfloat16','float32'):
            return 'Original Cdist owns FP16/BF16/FP32 input math; this integer/complex input has no audited lossless supported widening.'
        if op=='CountNonzeroFwdOp':
            dim=p.get('dim');axes=range(len(shape)) if dim is None else dim if isinstance(dim,(tuple,list)) else [dim]
            if prod(shape[a] for a in axes)>2**24:return 'FP32 count output cannot represent every integer in this reduction range exactly.'
    if op=='SubFwdOp' and p.get('alpha',1)!=1:return 'Original Sub graph has no alpha parameter; non-unit alpha requires additional arithmetic.'
    return None
