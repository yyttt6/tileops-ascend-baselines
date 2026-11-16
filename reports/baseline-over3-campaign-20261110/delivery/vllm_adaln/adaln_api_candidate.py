from pathlib import Path
from .runtime import validate,module,TRITON_ROOT,candidate
def adaln_original(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + 'fla/layernorm_guard.py')
    eps = case.params.get('eps', 1e-05)

    def run(x, scale, shift):
        n = x.shape[-1]
        xp, sp, bp = (a.contiguous().reshape(-1, n) for a in (x, scale, shift))
        out = [native._layer_norm_fwd(xp[i:i + 1], sp[i], bp[i], eps)[0] for i in range(len(xp))]
        return torch.cat(out).reshape(x.shape)
    return candidate(run, '_layer_norm_fwd (row-specific affine)', 'Per-row original LayerNorm calls select the corresponding manifest affine scale and shift; concatenate without host arithmetic.', [Path(__file__), *sources])
def adaln_fixed(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + 'fla/layernorm_guard.py')
    eps = case.params.get('eps', 1e-05)

    def run(x, scale, shift):
        n = x.shape[-1]
        xp, sp, bp = (a.contiguous().reshape(-1, n) for a in (x, scale, shift))
        out = native._layer_norm_fwd(xp.reshape(1, -1), sp.reshape(-1), bp.reshape(-1), eps, group_size=n)[0]
        return out.reshape(x.shape)
    return candidate(run, '_layer_norm_fwd (row-specific affine)', 'Per-row original LayerNorm calls select the corresponding manifest affine scale and shift; concatenate without host arithmetic.', [Path(__file__), *sources])
