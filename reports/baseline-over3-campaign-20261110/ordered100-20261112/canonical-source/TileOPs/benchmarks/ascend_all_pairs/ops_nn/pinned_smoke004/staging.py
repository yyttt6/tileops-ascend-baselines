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
