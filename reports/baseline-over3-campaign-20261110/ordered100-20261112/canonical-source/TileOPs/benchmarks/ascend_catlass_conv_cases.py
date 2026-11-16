"""Independent convolution groups via the original native convolution entry."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from benchmarks.ascend_baselines import UnsupportedCase, source_hashes


def catlass_conv_cases(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_catlass_baselines import catlass_conv
    from benchmarks.ascend_gemm_cases import catlass_gemm_cases
    from benchmarks.ascend_pointwise_conv import _no_bias_gemm

    x, weight, *extra = case.inputs
    params = case.params
    pair = lambda value: (value, value) if isinstance(value, int) else tuple(value)
    if (tuple(weight.shape[-2:]) == (1, 1) and params['groups'] == 1
            and pair(params['stride']) == (1, 1) and pair(params['padding']) == (0, 0)):
        native = _no_bias_gemm(case, catlass_gemm_cases)
        return replace(native, source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
    groups = params['groups']
    if groups == 1:
        native = catlass_conv(case)
        return replace(native, source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
    if any(t is not None for t in extra):
        raise UnsupportedCase('The original spatial convolution entry has no bias input.')
    cin, cout = x.shape[1] // groups, weight.shape[0] // groups
    def pack_weight(weight):
        return F.pad(weight, (0, 0, 0, 0, 0, 0, 0, -cout % 16))
    native = catlass_conv(SimpleNamespace(inputs=(x[:, :cin], pack_weight(weight[:cout])),
                                         params=params | {'groups': 1}))
    stride, padding, dilation = [pair(params[k]) for k in ('stride', 'padding', 'dilation')]
    spatial = [(x.shape[i + 2] + 2 * padding[i] - dilation[i] * (weight.shape[i + 2] - 1) - 1)
               // stride[i] + 1 for i in range(2)]
    def run(x, weight, bias=None):
        if bias is not None:
            raise UnsupportedCase("Original grouped convolution entry has no bias input.")
        out = torch.empty((x.shape[0], weight.shape[0], *spatial), device=x.device, dtype=x.dtype)
        for group in range(groups):
            value = native.implementation.run(x[:, group * cin:(group + 1) * cin],
                                              pack_weight(weight[group * cout:(group + 1) * cout]))
            # Native closure reuses storage: consume each result before next launch.
            out[:, group * cout:(group + 1) * cout].copy_(value[:, :cout])
        return out
    return replace(native, implementation=Implementation(run),
                   adaptation='Slice independent convolution groups; pad output-channel weights to16; invoke original complete convolution per group, crop/copy output channels; no split reduction or kernel/schedule changes.',
                   source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
