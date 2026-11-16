"""Scope-local lifetime adapters for exact SGL recovery keys.

These wrappers preserve the original SGL factories and kernels. They only keep
per-call tensor arguments alive, isolate async kernel inputs with contiguous
clones, and synchronize before returning so checker copy-back cannot race a
vendor launch. No arithmetic or kernel scheduling is changed.
"""
from dataclasses import replace

BMM_KEYS = {
    'mha-decode-pv-bfloat16', 'mha-decode-pv-float16',
    'mha-decode-qk-bfloat16', 'mha-decode-qk-float16',
    'mha-prefill-bfloat16', 'mha-prefill-float16',
    'mha-prefill-qk-bfloat16', 'mha-prefill-qk-float16',
    'mha-prefill-pv-bfloat16', 'mha-prefill-pv-float16',
    'mha-prefill-score-bfloat16',
}
PAGED_KEYS = {'ds-v3-mixed-bfloat16', 'llama-70b-cp-bfloat16'}

def _clone(value):
    import torch
    if isinstance(value, torch.Tensor):
        # Preserve dtype/shape/values while giving async vendor calls private
        # contiguous storage. Metadata tensors are normalized only in the
        # wrapper; the original candidate remains unchanged.
        return value.contiguous().clone()
    if isinstance(value, tuple): return tuple(_clone(v) for v in value)
    if isinstance(value, list): return [_clone(v) for v in value]
    if isinstance(value, dict): return {k: _clone(v) for k, v in value.items()}
    return value

def _adapt(candidate, label):
    import torch
    impl = candidate.implementation
    def run(*args, **kwargs):
        kept = _clone(args)
        kept_kwargs = _clone(kwargs)
        out = impl.run(*kept, **kept_kwargs)
        # Ensure all asynchronous vendor work has completed while `kept` is
        # still strongly referenced. This is a lifetime fix, not a fallback.
        torch.npu.synchronize()
        return out
    return replace(candidate, implementation=type(impl)(run, impl.args, impl.reset,
        impl.noncomparable_reason), adaptation=(candidate.adaptation +
        f'; scope-local async lifetime/contiguous-input adapter ({label})'))

def bmm(case):
    from .matmul import bmm as original
    return _adapt(original(case), 'Bmm exact SCOPE keys') if case.id in BMM_KEYS else original(case)

def paged_gather(case):
    from .gather import build as original
    return _adapt(original(case), 'PagedKVCacheGather exact SCOPE keys') if case.id in PAGED_KEYS else original(case)
def gdn(case):
    from .linear import gdn as original
    return _adapt(original(case), 'GDN exact SCOPE keys')
