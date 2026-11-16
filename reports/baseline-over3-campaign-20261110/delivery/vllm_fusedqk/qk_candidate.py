from pathlib import Path
from .runtime import TRITON_ROOT,candidate,module,validate
def qk_original(case):
    validate(case)
    import torch
    norm, sources = module(TRITON_ROOT + 'batch_invariant/rmsnorm.py')
    rotation, extra = module(TRITON_ROOT + 'rope.py', hardware=True)
    p = case.params

    def run(qkv, q_weight, k_weight, tables, positions):
        heads, kh, width = (p['num_heads'], p['num_kv_heads'], q_weight.numel())
        packed = qkv.clone().reshape(qkv.shape[0], heads + 2 * kh, width)
        parts = []
        for start, count, weight in ((0, heads, q_weight), (heads, kh, k_weight)):
            normalized = norm.rms_norm(packed[:, start:start + count].contiguous().reshape(-1, width).float(), weight.contiguous(), p['eps']).reshape(-1, count, width)
            parts.append(rotation.rope_forward_triton_siso(normalized, cos_sin_cache=tables.contiguous(), positions=positions.contiguous(), rope_dim=tables.shape[-1], is_neox_style=p['rope_layout'] == 'neox').to(qkv.dtype))
        return torch.cat((*parts, packed[:, heads + kh:]), dim=1).reshape(qkv.shape)
    return candidate(run, 'rms_norm + rope_forward_triton_siso (complete QKV)', 'Split Q/K/V layout and widen Q/K exactly to FP32; retain the original RMSNorm output in FP32 through original table-indexed RoPE, then round once to the input dtype. Return full QKV with V copied exactly and protect input QKV.', [Path(__file__), *sources, *extra])

def qk_fixed(case):
    validate(case)
    import torch
    norm, sources = module(TRITON_ROOT + 'fla/layernorm_guard.py')
    rotation, extra = module(TRITON_ROOT + 'rope.py', hardware=True)
    p = case.params

    def run(qkv, q_weight, k_weight, tables, positions):
        heads, kh, width = (p['num_heads'], p['num_kv_heads'], q_weight.numel())
        packed = qkv.clone().reshape(qkv.shape[0], heads + 2 * kh, width)
        parts = []
        for start, count, weight in ((0, heads, q_weight), (heads, kh, k_weight)):
            normalized = norm._layer_norm_fwd(packed[:, start:start + count].contiguous().reshape(-1, width).float(), weight.contiguous(), None, p['eps'], is_rms_norm=True)[0].reshape(-1, count, width)
            parts.append(rotation.rope_forward_triton_siso(normalized, cos_sin_cache=tables.contiguous(), positions=positions.contiguous(), rope_dim=tables.shape[-1], is_neox_style=p['rope_layout'] == 'neox').to(qkv.dtype))
        return torch.cat((*parts, packed[:, heads + kh:]), dim=1).reshape(qkv.shape)
    return candidate(run, 'rms_norm + rope_forward_triton_siso (complete QKV)', 'Split Q/K/V layout and widen Q/K exactly to FP32; retain the original RMSNorm output in FP32 through original table-indexed RoPE, then round once to the input dtype. Return full QKV with V copied exactly and protect input QKV.', [Path(__file__), *sources, *extra])
