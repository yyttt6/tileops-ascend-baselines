from math import prod
from pathlib import Path
from .runtime import validate,module,TRITON_ROOT,candidate
def group_norm_original(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + 'fla/layernorm_guard.py')
    p = case.params
    instance = case._call.signature.name == 'InstanceNormFwdOp'
    groups = p['x'][0][1] if instance else p['num_groups']
    eps = p.get('eps', 1e-05)

    def run(x, *args):
        if instance:
            _rm, _rv, weight, bias = args
        else:
            weight, bias = args
        batch, channels = x.shape[:2]
        spatial = prod(x.shape[2:])
        width = channels * spatial
        w = torch.ones(channels, device=x.device, dtype=x.dtype) if weight is None else weight
        w = w.reshape(channels, 1).expand(channels, spatial).contiguous().reshape(width)
        b = None if bias is None else bias.reshape(channels, 1).expand(channels, spatial).contiguous().reshape(width)
        packed = x.contiguous().reshape(batch, width)
        out = native._layer_norm_fwd(packed, w, b, eps, group_size=width // groups)[0]
        return out.reshape(x.shape)
    return candidate(run, '_layer_norm_fwd(group_size=group_elements)', 'Group contiguous channels/spatial elements; repeat per-channel affine weights over spatial positions each call. Input-statistics InstanceNorm is GroupNorm with one group per channel.', [Path(__file__), *sources])
def group_norm_fixed(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + 'fla/layernorm_guard.py')
    p = case.params
    instance = case._call.signature.name == 'InstanceNormFwdOp'
    groups = p['x'][0][1] if instance else p['num_groups']
    eps = p.get('eps', 1e-05)

    def run(x, *args):
        if instance:
            _rm, _rv, weight, bias = args
        else:
            weight, bias = args
        batch, channels = x.shape[:2]
        spatial = prod(x.shape[2:])
        width = channels * spatial
        w = torch.ones(channels, device=x.device, dtype=x.dtype) if weight is None else weight
        w = w.reshape(channels, 1).expand(channels, spatial).contiguous().reshape(width)
        b = None if bias is None else bias.reshape(channels, 1).expand(channels, spatial).contiguous().reshape(width)
        packed = x.contiguous().reshape(batch, width)
        out = native._layer_norm_fwd(packed, w, b, eps, group_size=width // groups)[0]
        return out.reshape(x.shape)
    return candidate(run, '_layer_norm_fwd(group_size=group_elements)', 'Group contiguous channels/spatial elements; repeat per-channel affine weights over spatial positions each call. Input-statistics InstanceNorm is GroupNorm with one group per channel.', [Path(__file__), *sources])
