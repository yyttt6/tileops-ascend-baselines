"""Exact-key routes for baseline-integration-700 nn_recovery01.

The delegated adapters are immutable byte copies under the private registration
receipt.  Loading is lazy so importing the provider remains CPU/device-free.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from benchmarks.ascend_baselines import UnsupportedCase

_ROOT = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/nn_registration01/frozen_adapters')
_KEYS = {
    ('FusedAddRMSNormFwdOp', 'llama-13b-prefill-bfloat16'),
    ('FusedAddRMSNormFwdOp', 'llama-405b-decode-bfloat16'),
    ('FusedAddRMSNormFwdOp', 'llama-405b-prefill-bfloat16'),
    ('FusedAddRMSNormFwdOp', 'llama-70b-decode-bfloat16'),
    ('FusedAddRMSNormFwdOp', 'llama-70b-prefill-bfloat16'),
    ('FusedAddRMSNormFwdOp', 'llama-8b-prefill-bfloat16'),
    ('FusedAddRMSNormFwdOp', 'llama-405b-prefill-float16'),
    ('FusedAddRMSNormFwdOp', 'llama-70b-prefill-float16'),
    ('FusedAddRMSNormFwdOp', 'llama-8b-prefill-float16'),
    ('GemmW4A16FwdOp', 'chunked-prefill-tile-float16'),
    ('GemmW4A16FwdOp', 'decode-b32-float16'),
    ('GemmW4A16FwdOp', 'decode-b8-float16'),
    ('FusedAddLayerNormFwdOp', 'llama-70b-decode-bfloat16'),
    ('FusedAddLayerNormFwdOp', 'llama-70b-prefill-bfloat16'),
    ('FusedAddLayerNormFwdOp', 'llama-8b-decode-bfloat16'),
    ('FusedAddLayerNormFwdOp', 'llama-8b-prefill-bfloat16'),
    ('FusedAddLayerNormFwdOp', 'llama-13b-prefill-float32'),
    ('GemmFP8FwdOp', 'ds-v3-m1-down-2d-float8_e4m3fn-bfloat16'),
}

def _adapter_for(op):
    name = 'adapter_3ab7708f1d.py' if op == 'FusedAddRMSNormFwdOp' else 'adapter_68ce198132.py'
    path = _ROOT / name
    spec = spec_from_file_location('integration700_nn_registration01_frozen_' + name[:10], path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def make(case):
    key = (case._call.signature.name, case.id)
    if key not in _KEYS:
        raise UnsupportedCase('No nn_registration01 exact receipt for this key')
    return _adapter_for(key[0]).make(case)
