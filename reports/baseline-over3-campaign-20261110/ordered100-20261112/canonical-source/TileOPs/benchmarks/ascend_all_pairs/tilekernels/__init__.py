"""TileKernels providers; package import never imports a device runtime."""
from typing import Any, Callable
from benchmarks.ascend_baselines import Candidate
from .adapters import silu_and_mul
from .extended import silu, mul, fused_add_rms, argmax, mhc_post, fp8, rope_variant
from benchmarks.ascend_baselines import tilekernels_add
from benchmarks.ascend_rms_cases import tilekernels_rms_cases
from benchmarks.ascend_reduction_cases import tilekernels_amax_cases
from benchmarks.ascend_rope_cases import tilekernels_rope_cases

PROVIDERS: dict[str, Callable[[Any], Candidate]] = {
    'AddFwdOp': tilekernels_add,
    'AmaxFwdOp': tilekernels_amax_cases,
    'ArgmaxFwdOp': argmax,
    'RMSNormFwdOp': tilekernels_rms_cases,
    'FusedAddRMSNormFwdOp': fused_add_rms,
    'SiluAndMulFwdOp': silu_and_mul,
    'SiluFwdOp': silu,
    'MulFwdOp': mul,
    'MHCPostFwdOp': mhc_post,
    'FP8QuantFwdOp': fp8,
    'RoPEFwdOp': tilekernels_rope_cases,
    'RoPELlama31FwdOp': rope_variant,
    'YaRNFwdOp': rope_variant,
    'LongRoPEFwdOp': rope_variant,
    'RoPENeoxPositionIdsFwdOp': rope_variant,
}

from .pointwise import neg, sub, sigmoid, maximum
PROVIDERS.update({"NegFwdOp": neg, "SubFwdOp": sub, "SigmoidFwdOp": sigmoid, "MaximumFwdOp": maximum})
from .pointwise import minimum, abs_value, argmin
PROVIDERS.update({'MinimumFwdOp': minimum, 'AbsFwdOp': abs_value, 'ArgminFwdOp': argmin})
from .reductions import sum_or_mean, minimum_or_boolean
PROVIDERS.update({'SumFwdOp': sum_or_mean, 'MeanFwdOp': sum_or_mean, 'AminFwdOp': minimum_or_boolean,
                  'AllFwdOp': minimum_or_boolean, 'AnyFwdOp': minimum_or_boolean})
from .pointwise import softplus
PROVIDERS['SoftplusFwdOp'] = softplus
from .dequant import int8_dequant
PROVIDERS.update({name: int8_dequant for name in ('INT8DequantPerTensorFwdOp', 'INT8DequantPerChannelFwdOp', 'INT8DequantPerBlockFwdOp')})

from .moe import fused_topk
from .selection import selection
from .additional import fused_qk_rope, roots, numerical_boolean
PROVIDERS['FusedTopKFwdOp'] = fused_topk
PROVIDERS['FusedQKNormRoPEFwdOp'] = fused_qk_rope
PROVIDERS.update({name: selection for name in (
    'ReluFwdOp', 'ClampScalarFwdOp', 'ClampTensorFwdOp', 'HardtanhFwdOp',
    'LeakyReluFwdOp', 'PreluFwdOp', 'HardsigmoidFwdOp', 'HardswishFwdOp',
    'LerpScalarFwdOp', 'LerpTensorFwdOp', 'LogicalAndFwdOp', 'LogicalOrFwdOp', 'LogicalNotFwdOp',
    'GeFwdOp', 'GtFwdOp', 'LeFwdOp', 'LtFwdOp', 'EqFwdOp', 'NeFwdOp')})
PROVIDERS.update({name: roots for name in ('RsqrtFwdOp', 'SqrtFwdOp', 'ReciprocalFwdOp', 'DivFwdOp')})
PROVIDERS.update({name: numerical_boolean for name in ('AllFwdOp', 'AnyFwdOp')})

from .pooling import pooling
PROVIDERS.update({name: pooling for name in ('MeanPoolingFwdOp', 'AvgPool1dFwdOp', 'AvgPool2dFwdOp', 'AvgPool3dFwdOp', 'MaxPool1dFwdOp', 'MaxPool2dFwdOp', 'MaxPool3dFwdOp', 'MaxPool1dIndicesFwdOp', 'MaxPool2dIndicesFwdOp', 'MaxPool3dIndicesFwdOp', 'AdaptiveAvgPool2dFwdOp', 'AdaptiveMaxPool2dFwdOp', 'AdaptiveMaxPool2dIndicesFwdOp')})

from .select_extra import extra
PROVIDERS.update({name: extra for name in ('CountNonzeroFwdOp', 'SignFwdOp', 'TanhFwdOp', 'WhereFwdOp', 'MaskedFillScalarFwdOp', 'MaskedFillTensorFwdOp')})

from .reciprocal import reciprocal
from .activation_composed import activation
PROVIDERS.update({name: reciprocal for name in ('ReciprocalFwdOp', 'DivFwdOp')})
PROVIDERS.update({name: activation for name in ('MishFwdOp', 'GeluTanhAndMulFwdOp')})

from .pooling_layout import pooling_layout
PROVIDERS.update({name: pooling_layout for name in tuple(PROVIDERS) if 'Pool' in name})

# Preserve the failed native composition for explicit diagnostics; do not register it as a baseline.
DIAGNOSTIC_PROVIDERS = {'FusedQKNormRoPEFwdOp': PROVIDERS.pop('FusedQKNormRoPEFwdOp')}

from .pooling_max import max_pool
PROVIDERS.update({name: max_pool for name in tuple(PROVIDERS) if 'MaxPool' in name})

# Resume22: preserve established small-real routes; full native sum for wide bool/int.
from .logical_complete import all_any, count_nonzero
PROVIDERS.update({'AllFwdOp': all_any, 'AnyFwdOp': all_any, 'CountNonzeroFwdOp': count_nonzero})

# Resume22 continuation: original native L1/L2 composition.
from .vector_norm import vector_norm
PROVIDERS['VectorNormFwdOp'] = vector_norm

# Resume22 continuation03: original native centered moments and normalization.
from .moments import moments
from .normalization import normalization
PROVIDERS.update({name: moments for name in ('VarFwdOp', 'StdFwdOp', 'VarMeanFwdOp')})
PROVIDERS.update({name: normalization for name in ('LayerNormFwdOp', 'FusedAddLayerNormFwdOp', 'GroupNormFwdOp', 'InstanceNormFwdOp')})
