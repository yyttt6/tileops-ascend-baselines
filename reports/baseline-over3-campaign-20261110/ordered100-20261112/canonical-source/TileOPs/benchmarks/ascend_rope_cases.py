"""Existing native NeoX/interleaved APIs with sequence-aware manifest layouts."""
from dataclasses import replace
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes
from benchmarks.ascend_transform_baselines import _rope_tables


def _layout(case):
    x, = case.inputs
    if x.ndim not in (2, 4) or x.shape[-1] not in (64, 128):
        raise UnsupportedCase('This adapter covers manifest [S,D] and [B,S,H,D] layouts.')
    if case.params['rope_layout'] not in ('neox', 'interleaved'):
        raise UnsupportedCase('Original native pairing must be NeoX or interleaved.')
    batch, seq, heads = (1, x.shape[0], 1) if x.ndim == 2 else x.shape[:3]
    return batch, seq, heads, x.shape[-1], case.params['rope_layout'] == 'neox'


def _tables(torch, x, base):
    # Positions restart at zero for each batch; heads share one sequence table.
    sequence = x if x.ndim == 2 else x[0, :, 0, :]
    return _rope_tables(torch, sequence, base)


def tilekernels_rope_cases(case):
    import torch
    from benchmarks.api import Implementation

    batch, seq, heads, dim, neox = _layout(case)
    host = 'tile_kernels/transform/rope_kernel.py'
    fn = local_function('TileKernels', '.', 'tile_kernels.transform.rope_kernel', 'apply_rotary', host)
    def run(x):
        cos, sin = _tables(torch, x, case.params.get('base', 10000.))
        cache = torch.cat((cos, sin), dim=-1).float()
        private = x.float().clone().reshape(batch, seq, heads, dim)
        positions = torch.arange(seq, device=x.device, dtype=torch.int32).expand(batch, seq).contiguous()
        fn(private, cache, positions=positions, interleaved=not neox)
        return private.reshape_as(x).to(x.dtype)
    root = LIBRARIES / 'TileKernels/tile_kernels'
    return Candidate(Implementation(run), 'TileKernels::apply_rotary',
                     'Native BSND and pairing flag; per-batch position reset; private exact-FP32 input and manifest-rounded tables; restore dtype/shape.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_transform_baselines.py'),
                                   root / 'transform/rope_kernel.py', root / 'transform/rope_asc.py', root / 'config.py'))


def sgl_rope_cases(case):
    import torch
    from benchmarks.api import Implementation

    batch, seq, heads, dim, neox = _layout(case)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_rope_qk_mqa.py'
    fn = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu',
                        'sgl_kernel_npu.norm.fused_rope_qk_mqa', 'fused_rope_qk_mqa', host)
    def run(x):
        cos, sin = _tables(torch, x, case.params.get('base', 10000.))
        cache = torch.cat((cos, sin), dim=-1).repeat(batch, 1).float()
        packed = x.float().contiguous().reshape(batch * seq, heads, dim)
        output, _ = fn(packed, packed, cache, dim, neox)
        return output.reshape_as(x).to(x.dtype)
    return Candidate(Implementation(run), 'sgl_kernel_npu::fused_rope_qk_mqa',
                     'Flatten batch/sequence; repeat manifest-rounded tables and widen exactly with input to FP32; original pairing flag and kernel; restore output shape/dtype.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_transform_baselines.py'),
                                   LIBRARIES / 'sgl-kernel-npu' / host))


def vllm_rope_cases(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_vllm_more import _original_module, _triton_bindings

    batch, seq, heads, dim, neox = _layout(case)
    host = 'vllm_ascend/ops/triton/rope.py'
    bindings = _triton_bindings()
    envs = _original_module('vllm_ascend/envs.py', {})
    utils = _original_module('vllm_ascend/ops/triton/triton_utils.py', bindings | {'envs': envs})
    utils.init_device_properties_triton()
    module = _original_module(host, bindings | {'get_ub_size_bytes': utils.get_ub_size_bytes,
                                               'get_vectorcore_num': utils.get_vectorcore_num})
    def run(x):
        cos, sin = _tables(torch, x, case.params.get('base', 10000.))
        private = x.clone().reshape(batch * seq, heads, dim)
        output = module.rope_forward_triton_siso(private, cos=cos.repeat(batch, 1), sin=sin.repeat(batch, 1),
                                                rope_dim=dim, is_neox_style=neox)
        return output.reshape_as(x)
    root = LIBRARIES / 'vllm-ascend'
    return Candidate(Implementation(run), 'vllm-ascend::rope_forward_triton_siso',
                     'Private input; flatten batch and sequence only; repeat rounded sequence tables per batch; original pairing flag.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_transform_baselines.py'),
                                   Path(__file__).with_name('ascend_vllm_more.py'), root / host,
                                   root / 'vllm_ascend/ops/triton/triton_utils.py', root / 'vllm_ascend/envs.py'))


def transformer_rope_cases(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_transformer_graph import _candidate

    batch, seq, heads, dim, neox = _layout(case)
    binding = GraphCall('ops-transformer')
    def run(x):
        tables = _tables(torch, x, case.params.get('base', 10000.))
        tables = [(torch.cat((t, t), dim=-1) if neox else t.repeat_interleave(2, dim=-1))
                  .float().reshape(1, seq, 1, dim) for t in tables]
        inp = x.float().contiguous().reshape(batch, seq, heads, dim)
        out = torch.empty_like(inp)
        binding.call('RotaryPositionEmbedding', [inp, *tables, None], [out], {'mode': 0 if neox else 1})
        return out.reshape_as(x).to(x.dtype)
    native = _candidate(binding, run, 'RotaryPositionEmbedding', 'posembedding/rotary_position_embedding',
                        'Native BSND table broadcasting and pairing mode; exact FP32 widening of input and manifest-rounded tables; output cast.')
    return replace(native, source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
