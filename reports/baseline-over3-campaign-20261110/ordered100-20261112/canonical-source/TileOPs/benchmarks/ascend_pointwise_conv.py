"""1x1 convolution expressed by layout-only adapters to original dense GEMMs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes


def _layout(case):
    x, weight, *extra = case.inputs
    p = case.params
    def pair(value):
        return (value, value) if isinstance(value, int) else tuple(value)
    if (tuple(weight.shape[-2:]) != (1, 1) or pair(p['stride']) != (1, 1)
            or pair(p['padding']) != (0, 0) or p['groups'] != 1):
        raise UnsupportedCase('Only ungrouped 1x1 stride1 pad0 convolution is a direct layout view of dense GEMM.')
    batch, channels, height, width = x.shape
    output_channels = weight.shape[0]
    def pack(x, weight):
        return (x.permute(0, 2, 3, 1).contiguous().reshape(-1, channels),
                weight.reshape(output_channels, channels))
    def unpack(value):
        return value.reshape(batch, height, width, output_channels).permute(0, 3, 1, 2).contiguous()
    return x, weight, extra, pack, unpack


def vllm_pointwise_conv(case):
    from benchmarks.api import Implementation
    from benchmarks.ascend_vllm_more import _original_module, _triton_bindings

    _, _, _, pack, unpack = _layout(case)
    host = 'vllm_ascend/ops/triton/batch_invariant/matmul.py'
    native = _original_module(host, _triton_bindings()).matmul_persistent
    def run(x, weight, bias=None):
        a, b = pack(x, weight)
        return unpack(native(a, b.T.contiguous(), bias=bias))
    return Candidate(Implementation(run), 'vllm-ascend::matmul_persistent (1x1 convolution)',
                     'NCHW to NHWC matrix view, transpose weights, original optional native bias, restore NCHW.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_vllm_more.py'), LIBRARIES / 'vllm-ascend' / host))


def _no_bias_gemm(case, factory, tile_shape=None):
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    x, weight, extra, pack, unpack = _layout(case)
    if any(t is not None for t in extra):
        raise UnsupportedCase('This original GEMM entry has no bias/accumulator input.')
    a, b = pack(x, weight)
    rows, channels = a.shape
    out_channels = b.shape[0]
    def pad(a, b):
        if tile_shape is None:
            return a, b
        bm, bn, bk = tile_shape
        return (F.pad(a, (0, -channels % bk, 0, -rows % bm)),
                F.pad(b, (0, -channels % bk, 0, -out_channels % bn)))
    native = factory(SimpleNamespace(inputs=pad(a, b), params={'trans_a': False, 'trans_b': True}))
    def run(x, weight, bias=None):
        a, b = pad(*pack(x, weight))
        return unpack(native.implementation.run(a, b)[:rows, :out_channels])
    return replace(native, implementation=Implementation(run),
                   adaptation='1x1 convolution: per-call NHWC matrix packing and weight transpose; zero-pad complete original tiles where required; crop and restore NCHW.',
                   source_sha256=native.source_sha256 | source_hashes(Path(__file__)))


def deepgemm_pointwise_conv(case):
    from benchmarks.ascend_baselines import deepgemm_gemm
    return _no_bias_gemm(case, deepgemm_gemm)


def tla_pointwise_conv(case):
    from benchmarks.ascend_tla_baselines import tla_gemm
    return _no_bias_gemm(case, tla_gemm, (128, 256, 64))


def transformer_pointwise_conv(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_transformer_graph import _candidate

    _, _, _, pack, unpack = _layout(case)
    binding = GraphCall('ops-transformer')
    def run(x, weight, bias=None):
        a, b = pack(x, weight)
        group = torch.full((1,), a.shape[1], device=x.device, dtype=torch.int64)
        shape = (1, a.shape[0], b.shape[0])
        out = (torch.zeros(shape, device=x.device, dtype=torch.float32) if bias is None
               else bias.to(torch.float32).reshape(1, 1, -1).expand(shape).contiguous())
        binding.call('GroupedMatmulAdd', [a.T.contiguous(), b.T.contiguous(), group, out], [out],
                     {'transpose_x': True, 'transpose_weight': False, 'group_type': 2, 'group_list_type': 0})
        return unpack(out[0].to(x.dtype))
    native = _candidate(binding, run, 'GroupedMatmulAdd', 'gmm/grouped_matmul_add',
                        '1x1 convolution: NHWC matrix packing, transpose native GEMM inputs; optional bias initializes native FP32 accumulator; restore NCHW.')
    return replace(native, source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
