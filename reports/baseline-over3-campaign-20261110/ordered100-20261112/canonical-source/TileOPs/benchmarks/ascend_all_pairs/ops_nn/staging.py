"""Implemented candidates awaiting per-case validation; not the public registry."""
from .catalog import UNARY,FOREACH,POOL,NORM,LEGACY
from .elementwise import activation,foreach,softmax,dequant
from .pooling import pool,reduction_pool
from .normalization import normalization
from .matrix import bmm,convolution,legacy

CANDIDATES={**{name:activation for name in UNARY},**{name:foreach for name in FOREACH},
            **{name:pool for name in POOL},**{name:normalization for name in NORM},
            **{name:legacy for name in LEGACY},
            'SoftmaxFwdOp':softmax,'LogSoftmaxFwdOp':softmax,'BmmFwdOp':bmm,
            'Conv1dFwdOp':convolution,'Conv3dFwdOp':convolution,
            'AmaxFwdOp':reduction_pool,'ArgmaxFwdOp':reduction_pool,'MeanFwdOp':reduction_pool,
            **{f'INT8DequantPer{granularity}FwdOp':dequant for granularity in ('Tensor','Channel','Block')}}
from .elementwise import clamp
from .pooling import chunk_mean
from .normalization import count_nonzero
CANDIDATES.update({'CountNonzeroFwdOp':count_nonzero,'MeanPoolingFwdOp':chunk_mean,
                  **{name:clamp for name in ('ClampScalarFwdOp','ClampTensorFwdOp','HardtanhFwdOp')}})
from .matrix import grouped,fp8_gemm
CANDIDATES.update({'GroupedGemmFwdOp':grouped,'GemmFP8FwdOp':fp8_gemm,'BmmFP8FwdOp':fp8_gemm})
from .normalization import variance
CANDIDATES.update({name:variance for name in ('VarFwdOp','StdFwdOp','VarMeanFwdOp')})
from .reductions import segment,argmin,rsqrt
CANDIDATES.update({name:segment for name in ('SumFwdOp','ProdFwdOp','AminFwdOp','AllFwdOp','AnyFwdOp')})
CANDIDATES.update({'ArgminFwdOp':argmin,'RsqrtFwdOp':rsqrt})
from .matrix import moe_grouped
CANDIDATES['MoEGroupedGemmFwdOp']=moe_grouped
from .matrix import w4_gemm,moe_fp8_grouped
CANDIDATES.update({'GemmW4A16FwdOp':w4_gemm,'MoEGroupedGemmFP8FwdOp':moe_fp8_grouped})
from .elementwise import masked_fill
CANDIDATES.update({name:masked_fill for name in ('MaskedFillScalarFwdOp','MaskedFillTensorFwdOp')})
from .matrix import int8_gemm
CANDIDATES['GemmINT8W8A8FwdOp']=int8_gemm
from .normalization import ada_layer_norm
CANDIDATES.update({name:ada_layer_norm for name in ('AdaLayerNormFwdOp','AdaLayerNormZeroFwdOp')})
from .matrix import causal_convolution
CANDIDATES.update({name:causal_convolution for name in ('CausalConv1dFwdOp','CausalConv1dUpdateFwdOp')})
from .elementwise import boolean_logic
from .catalog import BOOLEAN_NAMES
CANDIDATES.update({name:boolean_logic for name in BOOLEAN_NAMES})
from .elementwise import where
CANDIDATES['WhereFwdOp']=where
from .cache import cache_gather,cache_write
CANDIDATES['PagedKVCacheGatherFwdOp']=cache_gather
CANDIDATES.update({name:cache_write for name in ('PagedKVCacheWriteFwdOp','MLAKVCacheWriteFwdOp')})
from .elementwise import bitwise_not
CANDIDATES['BitwiseNotFwdOp']=bitwise_not
from .wide_truth import dispatch as wide_truth_dispatch
CANDIDATES.update({name:wide_truth_dispatch for name in ('AllFwdOp','AnyFwdOp','CountNonzeroFwdOp')})
from .bitwise_bytes import dispatch as bitwise_bytes_dispatch
CANDIDATES.update({name:bitwise_bytes_dispatch for name in ('BitwiseAndFwdOp','BitwiseOrFwdOp','BitwiseXorFwdOp')})
from .ieee_lookup import factory as ieee_lookup_factory
CANDIDATES.update({name:ieee_lookup_factory for name in ('IsfiniteFwdOp','IsinfFwdOp','IsnanFwdOp','NanToNumFwdOp')})
from .adaptive_max_lift import factory as adaptive_max_lift_factory
CANDIDATES.update({name:adaptive_max_lift_factory for name in ('AdaptiveMaxPool2dFwdOp','AdaptiveMaxPool2dIndicesFwdOp')})
from .logsumexp import factory as logsumexp_factory
CANDIDATES['LogSumExpFwdOp']=logsumexp_factory

from .mean_axis_lift import factory as mean_axis_lift_factory
CANDIDATES['MeanFwdOp']=mean_axis_lift_factory

from .conv2d_native import factory as conv2d_native_factory
CANDIDATES['Conv2dFwdOp']=conv2d_native_factory

# Retire only demonstrated composed entrypoints; keep historical private copies.
for _retired_name in ('LogicalAndFwdOp', 'LogicalOrFwdOp', 'LogicalNotFwdOp', 'BitwiseNotFwdOp', 'StdFwdOp', 'AdaLayerNormFwdOp', 'AdaLayerNormZeroFwdOp', 'FusedAddRMSNormFwdOp', 'FusedAddLayerNormFwdOp'):
    CANDIDATES.pop(_retired_name, None)


def _original_variance(case):
    if int(case.params.get('correction', 1)) != 0:
        raise RuntimeError('Retired variance correction composition; original zero-correction auxiliary-output route remains available.')
    return variance(case)


def _original_clamp(case):
    from .common import op_name
    name = op_name(case)
    if name == 'HardtanhFwdOp' or (case.params.get('min') is not None and case.params.get('max') is not None):
        raise RuntimeError('Retired two-bound max/min composition; single-bound original API routes remain available.')
    return clamp(case)


CANDIDATES.update({name: _original_variance for name in ('VarFwdOp', 'VarMeanFwdOp')})
CANDIDATES.update({name: _original_clamp for name in ('ClampScalarFwdOp', 'ClampTensorFwdOp', 'HardtanhFwdOp')})
