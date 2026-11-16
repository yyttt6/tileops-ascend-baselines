"""Lazy ops-math providers. Importing this module never imports a device runtime."""
from functools import partial
from .specs import SPECS

def _factory(case, op):
    from .graph import make
    return make(case,op)

def _add(case):
    from benchmarks.ascend_math_baselines import opsmath_add
    return opsmath_add(case)

CANDIDATES = {op:partial(_factory,op=op) for op in SPECS}
CANDIDATES['AddFwdOp']=_add

def _verified_abs(case):
    from .validated_abs.graph import make
    return make(case,'AbsFwdOp')

CANDIDATES['AbsFwdOp']=_verified_abs

def _norm(case,op):
    if op=='BatchNormFwdOp' and (not case.params.get('training',True) or case.params.get('running_mean') is not None):
        from .running_stats import make
        return make(case,op)
    from .layernorm import make
    return make(case,op)

for _op in ('LayerNormFwdOp','AdaLayerNormFwdOp','GroupNormFwdOp','InstanceNormFwdOp','BatchNormFwdOp'):
    CANDIDATES[_op]=partial(_norm,op=_op)

def _pool(case):
    from .mean_pooling import make
    return make(case)

CANDIDATES['MeanPoolingFwdOp']=_pool

def _window_pool(case,op):
    from .pooling import make
    return make(case,op)

for _op in ('AvgPool1dFwdOp','AvgPool2dFwdOp','AvgPool3dFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveAvgPool2dFwdOp','AdaptiveMaxPool2dFwdOp'):
    CANDIDATES[_op]=partial(_window_pool,op=_op)

def _weighted(case,op):
    from .weighted import make
    return make(case,op)

for _op in ('GemmFwdOp','BmmFwdOp','GroupedGemmFwdOp','Conv1dFwdOp','Conv2dFwdOp','Conv3dFwdOp','SSDChunkCouplingFwdOp','MoEGroupedGemmFwdOp','MHCPostFwdOp'):
    CANDIDATES[_op]=partial(_weighted,op=_op)

def _dequant(case,op):
    from .dequant import make
    return make(case,op)

for _op in ('INT8DequantPerTensorFwdOp','INT8DequantPerChannelFwdOp','INT8DequantPerBlockFwdOp'):
    CANDIDATES[_op]=partial(_dequant,op=_op)

def _distance(case,op):
    from .distance import make
    return make(case,op)

for _op in ('VectorNormFwdOp','CountNonzeroFwdOp'):
    CANDIDATES[_op]=partial(_distance,op=_op)

def _pool_indices(case,op):
    from .pool_indices import make
    return make(case,op)

for _op in ('MaxPool1dIndicesFwdOp','MaxPool2dIndicesFwdOp','MaxPool3dIndicesFwdOp','AdaptiveMaxPool2dIndicesFwdOp'):
    CANDIDATES[_op]=partial(_pool_indices,op=_op)

def _topk_select(case):
    from .topk_select import make
    return make(case)

CANDIDATES['TopKSelectFwdOp']=_topk_select

# Original-only boundary: mathematical compositions are retired from live routes.
# AdaLayerNorm retains the existing original LayerNorm row API registered above.
# Historical composition.py and receipts remain intact for provenance.

# Public registration contains accepted executable routes; trials stay private.
import json as _json
from pathlib import Path as _Path
from .semantic_limits import reason as _semantic_reason
NATIVE_CANDIDATES=dict(CANDIDATES)

def _screened(case,op,factory):
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
    reason=_semantic_reason(op,case.params)
    if reason:raise UnsupportedCase(reason)
    if op=='AddFwdOp':
        candidate=factory(case)
    else:
        from .isolated import make
        candidate=make(case,op)
    candidate.source_sha256.update(source_hashes(*(_Path(__file__).with_name(name) for name in
        ('__init__.py','semantic_limits.py','specs.py','composition_specs.py'))))
    return candidate

CANDIDATES={op:partial(_screened,op=op,factory=factory) for op,factory in CANDIDATES.items()}
_enabled = _json.loads(_Path(__file__).with_name("enabled.json").read_text())
_enabled_cases = _json.loads(_Path(__file__).with_name("enabled_cases.json").read_text())

def _public(case,op):
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.id not in _enabled_cases.get(op,()):
        raise UnsupportedCase('This ops-math case has no accepted complete numerical and native-ownership receipt.')
    return CANDIDATES[op](case)

PROVIDERS = {op: partial(_public,op=op) for op in _enabled}

# Exact full original API receipt: baseline700/fft_window_first1 (system AiCPU STFT).
from . import integration700_fft_window_first1 as _fft_window_first1
_fft_window_first1_prior = PROVIDERS.get('FFTC2CFwdOp')
def _fft_window_first1_route(case, _prior=_fft_window_first1_prior):
    if (case._call.signature.name, case.id) == _fft_window_first1._KEY:
        return _fft_window_first1.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No exact FFT route')
PROVIDERS['FFTC2CFwdOp'] = _fft_window_first1_route

# Original complete aclStft receipts: fft_remaining_c128_2; system CANN AiCPU.
from . import integration700_fft_remaining_c128_2 as _fft_remaining_c128_2
_fft_remaining_c128_2_prior = PROVIDERS.get('FFTC2CFwdOp')
def _fft_remaining_c128_2_route(case, _prior=_fft_remaining_c128_2_prior):
    if (case._call.signature.name, case.id) in _fft_remaining_c128_2._KEYS:
        return _fft_remaining_c128_2.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No exact FFT route')
PROVIDERS['FFTC2CFwdOp'] = _fft_remaining_c128_2_route

# Original complete aclStft receipts: fft_remaining_c64_7; system CANN AiCPU.
from . import integration700_fft_remaining_c64_7 as _fft_remaining_c64_7
_fft_remaining_c64_7_prior = PROVIDERS.get('FFTC2CFwdOp')
def _fft_remaining_c64_7_route(case, _prior=_fft_remaining_c64_7_prior):
    if (case._call.signature.name, case.id) in _fft_remaining_c64_7._KEYS:
        return _fft_remaining_c64_7.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No exact FFT route')
PROVIDERS['FFTC2CFwdOp'] = _fft_remaining_c64_7_route


# baseline700: one exact original Mul dequant receipt; preserve previous factory.
from . import integration700_nn_dequant_registration1 as _nn_dequant_registration1
_nn_dequant_registration1_previous = PROVIDERS.get('INT8DequantPerChannelFwdOp')
def _nn_dequant_registration1_route(case, _previous=_nn_dequant_registration1_previous):
    if (case._call.signature.name, case.id) == _nn_dequant_registration1.KEY[1:]:
        # Same key with wrong params must fail closed, not fall through to old route.
        return _nn_dequant_registration1.make(case, baseline='ops-math')
    if _previous is not None:
        return _previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact dequant route')
PROVIDERS['INT8DequantPerChannelFwdOp'] = _nn_dequant_registration1_route

# Single-agent original ReduceSum with isolated deployment.
from .single_agent_local_sum import install as _install_single_local_sum
_install_single_local_sum(PROVIDERS)
