from math import prod
from pathlib import Path
from .runtime import TRITON_ROOT,candidate,module,validate
def normalization_original(case):
    validate(case)
    import torch
    op = case._call.signature.name
    rms = op in ('RMSNormFwdOp', 'FusedAddRMSNormFwdOp')
    fused = op.startswith('FusedAdd')
    native, sources = module(TRITON_ROOT + ('batch_invariant/rmsnorm.py' if rms else 'fla/layernorm_guard.py'))
    if fused:
        addition, add_sources = module(TRITON_ROOT + 'muls_add.py', hardware=True)
        sources += add_sources
    p = case.params
    eps = p.get('eps')
    eps = 2 ** (-23) if eps is None else eps
    hidden = prod(p.get('normalized_shape', [p['x'][0][-1]]))

    def normalize(x, w, b=None):
        packed = x.contiguous().reshape(-1, hidden)
        weight = torch.ones(hidden, device=x.device, dtype=x.dtype) if w is None else w.contiguous().reshape(hidden)
        if rms:
            return native.rms_norm(packed, weight, eps).reshape(x.shape)
        bias = b.contiguous().reshape(hidden) if b is not None else None
        return native._layer_norm_fwd(packed, weight, bias, eps)[0].reshape(x.shape)

    def run(x, *args):
        if fused:
            residual, w, *bias = args
            summed = addition.muls_add_triton(x.contiguous().reshape(-1, 1), residual.contiguous().reshape(-1, 1), 1.0).reshape(x.shape)
            return (normalize(summed, w, bias[0] if bias else None), summed)
        w, *bias = args
        return normalize(x, w, bias[0] if bias else None)
    return candidate(run, ('muls_add_triton + ' if fused else '') + ('rms_norm' if rms else '_layer_norm_fwd'), 'Collapse normalized dimensions; native weighted normalization; optional affine disabled with unit weight. Fused cases first call original native addition and retain its input-dtype rounded sum as the second output.', [Path(__file__), *sources])
def normalization_fixed(case):
    validate(case)
    import torch
    op = case._call.signature.name
    rms = op in ('RMSNormFwdOp', 'FusedAddRMSNormFwdOp')
    fused = op.startswith('FusedAdd')
    native, sources = module(TRITON_ROOT + ('fla/layernorm_guard.py'))
    if fused:
        addition, add_sources = module(TRITON_ROOT + 'muls_add.py', hardware=True)
        sources += add_sources
    p = case.params
    eps = p.get('eps')
    eps = 2 ** (-23) if eps is None else eps
    hidden = prod(p.get('normalized_shape', [p['x'][0][-1]]))

    def normalize(x, w, b=None):
        packed = x.contiguous().reshape(-1, hidden)
        weight = torch.ones(hidden, device=x.device, dtype=x.dtype) if w is None else w.contiguous().reshape(hidden)
        if rms:
            return native._layer_norm_fwd(packed, weight, None, eps, is_rms_norm=True)[0].reshape(x.shape)
        bias = b.contiguous().reshape(hidden) if b is not None else None
        return native._layer_norm_fwd(packed, weight, bias, eps)[0].reshape(x.shape)

    def run(x, *args):
        if fused:
            residual, w, *bias = args
            summed = addition.muls_add_triton(x.contiguous().reshape(-1, 1), residual.contiguous().reshape(-1, 1), 1.0).reshape(x.shape)
            return (normalize(summed, w, bias[0] if bias else None), summed)
        w, *bias = args
        return normalize(x, w, bias[0] if bias else None)
    return candidate(run, ('muls_add_triton + ' if fused else '') + ('rms_norm' if rms else '_layer_norm_fwd'), 'Collapse normalized dimensions; native weighted normalization; optional affine disabled with unit weight. Fused cases first call original native addition and retain its input-dtype rounded sum as the second output.', [Path(__file__), *sources])
