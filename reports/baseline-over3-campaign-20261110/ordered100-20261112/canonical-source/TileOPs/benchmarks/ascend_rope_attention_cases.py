"""Original native RoPE followed by the same library's native attention API."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from benchmarks.ascend_baselines import LIBRARIES, UnsupportedCase, local_function, source_hashes


def _rotation(library, neox, rotary_dim):
    import torch

    if library == 'sgl-kernel-npu':
        host = 'python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_rope_qk_mqa.py'
        fn = local_function(library, 'python/sgl_kernel_npu',
                            'sgl_kernel_npu.norm.fused_rope_qk_mqa', 'fused_rope_qk_mqa', host)

        def rotate(x, cos, sin):
            batch, seq, heads, dim = x.shape
            data = x.float().contiguous().reshape(batch * seq, heads, dim)
            tables = torch.cat((cos, sin), dim=-1).repeat(batch, 1).float()
            out, _ = fn(data, data, tables, rotary_dim, neox)
            return out.reshape_as(x).to(x.dtype)

        return rotate, source_hashes(LIBRARIES / library / host), None

    if library == 'vllm-ascend':
        from benchmarks.ascend_vllm_more import _original_module, _triton_bindings

        bindings = _triton_bindings()
        envs = _original_module('vllm_ascend/envs.py', {})
        utils = _original_module('vllm_ascend/ops/triton/triton_utils.py', bindings | {'envs': envs})
        utils.init_device_properties_triton()
        host = 'vllm_ascend/ops/triton/rope.py'
        module = _original_module(host, bindings | {'get_ub_size_bytes': utils.get_ub_size_bytes,
                                                    'get_vectorcore_num': utils.get_vectorcore_num})

        def rotate(x, cos, sin):
            batch, seq, heads, dim = x.shape
            data = x.float().contiguous().reshape(batch * seq, heads, dim)
            out = module.rope_forward_triton_siso(
                data, cos=cos.repeat(batch, 1).float(), sin=sin.repeat(batch, 1).float(),
                rope_dim=rotary_dim, is_neox_style=neox)
            return out.reshape_as(x).to(x.dtype)

        root = LIBRARIES / library
        return rotate, source_hashes(Path(__file__).with_name('ascend_vllm_more.py'), root / host,
                                     root / 'vllm_ascend/ops/triton/triton_utils.py',
                                     root / 'vllm_ascend/envs.py'), None

    if library == 'ops-transformer':
        from benchmarks.ascend_graph_call import GraphCall
        from benchmarks.ascend_transformer_graph import _candidate

        binding = GraphCall(library)

        def rotate(x, cos, sin):
            seq = x.shape[1]
            tables = [(torch.cat((t, t), dim=-1) if neox else t.repeat_interleave(2, dim=-1))
                      .float().reshape(1, seq, 1, rotary_dim) for t in (cos, sin)]
            part = x[..., :rotary_dim].float().contiguous()
            output = torch.empty_like(part)
            binding.call('RotaryPositionEmbedding', [part, *tables, None], [output],
                         {'mode': 0 if neox else 1})
            return torch.cat((output.to(x.dtype), x[..., rotary_dim:]), dim=-1).contiguous()

        candidate = _candidate(binding, rotate, 'RotaryPositionEmbedding',
                               'posembedding/rotary_position_embedding',
                               'Use supplied manifest tables and native pairing, preserve unrotated channels.')
        return rotate, candidate.source_sha256, candidate.cleanup
    raise UnsupportedCase('No reviewed original rotation entry for this library.')


def _with_rope(case, library, attention_factory):
    from benchmarks.api import Implementation

    if case.params.get('pos_encoding_mode', 'none') == 'none':
        return attention_factory(case)
    if len(case.inputs) != 8:
        raise UnsupportedCase('Expected original dense GQA input order including optional scales and tables.')
    q, k, v, qs, ks, vs, cos, sin = case.inputs
    p = case.params
    if (any(t is not None for t in (qs, ks, vs)) or cos is None or sin is None
            or p.get('pos_encoding_mode') != 'rope' or p.get('rope_layout') not in ('neox', 'interleaved')
            or p.get('rotary_dim') not in (64, 128) or q.shape[-1] < p['rotary_dim']
            or tuple(cos.shape) != (k.shape[1], p['rotary_dim'] // 2) or sin.shape != cos.shape):
        raise UnsupportedCase('Requires native partial/full RoPE with manifest cos/sin and no quantization scales.')
    params = dict(p, pos_encoding_mode='none', rotary_dim=None)
    native = attention_factory(SimpleNamespace(inputs=(q, k, v, None, None, None, None, None),
                                                params=params))
    rotate, rotation_hashes, rotation_cleanup = _rotation(library, p['rope_layout'] == 'neox', p['rotary_dim'])

    def run(q, k, v, qs, ks, vs, cos, sin):
        # Positions are right-aligned for Q and start at zero for K. Every call
        # consumes current table/input values; no numerical results are cached.
        offset = k.shape[1] - q.shape[1]
        query = rotate(q, cos[offset:], sin[offset:])
        key = rotate(k, cos, sin)
        return native.implementation.run(query, key, v, None, None, None, None, None)

    def cleanup():
        try:
            if native.cleanup is not None:
                native.cleanup()
        finally:
            if rotation_cleanup is not None:
                rotation_cleanup()

    return replace(native, implementation=Implementation(run),
                   cleanup=cleanup,
                   api=library + '::native RoPE + native attention',
                   adaptation='Original native rotation of the first rotary_dim channels using supplied tables, then original native attention with existing causal mask. Layout/casts only around unchanged native kernels; all stages count in execution time.',
                   source_sha256=native.source_sha256 | rotation_hashes | source_hashes(Path(__file__)),
                   limitations=(*native.limitations, 'Both rotation launches and attention are included; native graph ownership must be checked for both Transformer operators.'))


def sgl_rope_gqa_cases(case):
    from benchmarks.ascend_attention_cases import sgl_gqa_cases
    return _with_rope(case, 'sgl-kernel-npu', sgl_gqa_cases)


def vllm_rope_gqa_cases(case):
    from benchmarks.ascend_attention_cases import vllm_gqa_cases
    return _with_rope(case, 'vllm-ascend', vllm_gqa_cases)


def transformer_rope_gqa_cases(case):
    from benchmarks.ascend_attention_cases import transformer_gqa_cases
    return _with_rope(case, 'ops-transformer', transformer_gqa_cases)
