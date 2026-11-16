"""Isolated ops-transformer providers; importing this module is device-free."""
from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.ascend_baselines import Candidate

from .adapters import bmm, layernorm, pointwise_conv, segmented_attention, mhc_post, softmax, postpermute, cache_write

PROVIDERS: dict[str, Callable[[Any], Candidate]] = {
    'BmmFwdOp': bmm, 'LayerNormFwdOp': layernorm,
    'Conv1dFwdOp': pointwise_conv, 'Conv3dFwdOp': pointwise_conv,
    'GQAVarlenFwdOp': segmented_attention, 'GQAPagedFwdOp': segmented_attention,
    'MHCPostFwdOp': mhc_post, 'SoftmaxFwdOp': softmax,
    'MoEPostPermuteFwdOp': postpermute, 'PagedKVCacheWriteFwdOp': cache_write,
}

from .adapters import recurrent_delta, fused_topk, reduction, grouped_gemm
PROVIDERS.update({
    'DeltaNetRecurrentFwdOp': recurrent_delta, 'FusedTopKFwdOp': fused_topk,
    'SumFwdOp': reduction, 'MeanFwdOp': reduction,
    'GroupedGemmFwdOp': grouped_gemm, 'MoEGroupedGemmFwdOp': grouped_gemm,
})

def _existing(op):
    def factory(case):
        from benchmarks.ops.bench_ascend_classics import PROVIDERS as classic
        return classic[op]['ops-transformer'](case)
    return factory

# Existing successful routes are imported only inside a device reservation.
for _op in ('AddFwdOp','AmaxFwdOp','GemmFwdOp','RMSNormFwdOp','RoPEFwdOp','MoEPrePermuteFwdOp','Conv2dFwdOp','GQADenseFwdOp'):
    PROVIDERS[_op]=_existing(_op)
from .adapters import rope_variant, coupling
PROVIDERS.update({op:rope_variant for op in ('RoPELlama31FwdOp','YaRNFwdOp','LongRoPEFwdOp','RoPENeoxPositionIdsFwdOp')})
PROVIDERS['SSDChunkCouplingFwdOp']=coupling
from .adapters import gqa_backward, chunk_delta, cache_gather
PROVIDERS.update({'GQABwdOp':gqa_backward,'DeltaNetChunkFwdOp':chunk_delta,'DeltaNetFwdOp':chunk_delta,'GDNFwdOp':chunk_delta,'PagedKVCacheGatherFwdOp':cache_gather})
from .adapters import causal_conv, recurrent_family
PROVIDERS.update({op:causal_conv for op in ('CausalConv1dFwdOp','CausalConv1dUpdateFwdOp')})
PROVIDERS.update({op:recurrent_family for op in ('KDAFwdOp','DeltaNetFwdOp','GDNFwdOp','DeltaNetRecurrentFwdOp')})

from .adapters import channel_norm
PROVIDERS.update({op:channel_norm for op in ('GroupNormFwdOp','InstanceNormFwdOp')})

from .adapters import gelu_tanh_mul, silu
PROVIDERS.update({'GeluTanhAndMulFwdOp':gelu_tanh_mul,'SiluFwdOp':silu})

from .adapters import quant_gemm
PROVIDERS.update({op:quant_gemm for op in ('GemmFP8FwdOp','BmmFP8FwdOp','GemmINT8W8A8FwdOp')})
# No manifest case is executable for these original contracts.
for _op in ('KDAFwdOp','LayerNormFwdOp','GroupNormFwdOp','DeltaNetChunkFwdOp'):
    PROVIDERS.pop(_op,None)

from .adapters import decode_attention
PROVIDERS.update({op:decode_attention for op in ('MLADecodeWithKVCacheFwdOp','MHADecodePagedWithKVCacheFwdOp')})

from .adapters import dsa
PROVIDERS['DSADecodeWithKVCacheFwdOp']=dsa

from .adapters import state_passing
PROVIDERS['SSDStatePassingFwdOp']=state_passing
from .adapters import prefill_append,nsa_selected
PROVIDERS.update({'GQAPrefillPagedWithKVCacheFwdOp':prefill_append,'NSAVarlenFwdOp':nsa_selected})
from .adapters import nsa_compressed
PROVIDERS['NSACompressedVarlenFwdOp']=nsa_compressed
from .adapters import mla_cache_write,mla_varlen
PROVIDERS.update({'MLAKVCacheWriteFwdOp':mla_cache_write,'MLAVarlenFwdOp':mla_varlen})
from .adapters import fused_qk_rope
PROVIDERS['FusedQKNormRoPEFwdOp']=fused_qk_rope
from .adapters import topk_select
PROVIDERS['TopKSelectFwdOp']=topk_select
from .adapters import fp8_block_gemm,fp8_gemm_dispatch
PROVIDERS.update({'GemmFP8FwdOp':fp8_gemm_dispatch,'MoEGroupedGemmFP8FwdOp':fp8_block_gemm})
from .adapters import merge_attention
PROVIDERS['MergeAttentionStatesFwdOp']=merge_attention
from .adapters import chunk_scan
PROVIDERS['SSDChunkScanFwdOp']=chunk_scan
from .adapters import w4a16
PROVIDERS['GemmW4A16FwdOp']=w4a16
from .adapters import hadamard
PROVIDERS['HadamardTransformFwdOp']=hadamard

from .mla import paged as mla_paged
PROVIDERS['MLAPagedFwdOp']=mla_paged

from .conv import conv2d
PROVIDERS['Conv2dFwdOp']=conv2d

from .elementwise import sigmoid
PROVIDERS['SigmoidFwdOp']=sigmoid

from .linear import linear,reduction_dispatch
PROVIDERS.update({'NegFwdOp':linear,'SubFwdOp':linear,'SumFwdOp':reduction_dispatch,'MeanFwdOp':reduction_dispatch})

from .log_reduction import log_reduction
PROVIDERS.update({op:log_reduction for op in ('LogSumExpFwdOp','LogSoftmaxFwdOp')})

from .pooling import pooling
PROVIDERS.update({op:pooling for op in ('AvgPool1dFwdOp', 'AvgPool2dFwdOp', 'AvgPool3dFwdOp', 'AdaptiveAvgPool2dFwdOp', 'MeanPoolingFwdOp')})

from .multiply import multiply
PROVIDERS['MulFwdOp']=multiply

from .routing import routing
PROVIDERS['FusedTopKFwdOp']=routing

from .dsa_paged import dsa_paged
PROVIDERS['DSAPagedFwdOp']=dsa_paged

from .extrema import extrema
PROVIDERS.update({op:extrema for op in ('AmaxFwdOp','AminFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveMaxPool2dFwdOp')})
from .topk import topk_select as hierarchical_topk
PROVIDERS['TopKSelectFwdOp']=hierarchical_topk

from .pointwise_extrema import pointwise_extrema
PROVIDERS.update({op:pointwise_extrema for op in ('AbsFwdOp','MaximumFwdOp','MinimumFwdOp','ReluFwdOp','ClampScalarFwdOp','ClampTensorFwdOp','HardtanhFwdOp')})
from .prefix_sum import prefix_sum
PROVIDERS['CumsumFwdOp']=prefix_sum
PROVIDERS['LerpScalarFwdOp']=linear
from .arg_indices import arg_indices
PROVIDERS.update({'ArgmaxFwdOp':arg_indices,'ArgminFwdOp':arg_indices})

from .pool_indices import pool_indices
PROVIDERS.update({op:pool_indices for op in ('MaxPool1dIndicesFwdOp','MaxPool2dIndicesFwdOp','MaxPool3dIndicesFwdOp','AdaptiveMaxPool2dIndicesFwdOp')})

from .exponential import exponential
PROVIDERS.update({'ExpFwdOp':exponential,'Expm1FwdOp':exponential})

from .selection_values import extrema_dispatch, pointwise_dispatch
PROVIDERS.update({op:extrema_dispatch for op in ('AmaxFwdOp','AminFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveMaxPool2dFwdOp')})
PROVIDERS.update({op:pointwise_dispatch for op in ('AbsFwdOp','MaximumFwdOp','MinimumFwdOp','ReluFwdOp','ClampScalarFwdOp','ClampTensorFwdOp','HardtanhFwdOp')})

from .gating_normalization import softmax_dispatch, sigmoid_selection, fused_topk_dispatch
PROVIDERS.update({"SoftmaxFwdOp": softmax_dispatch, "SigmoidFwdOp": sigmoid_selection, "FusedTopKFwdOp": fused_topk_dispatch})

from .routing_arithmetic import dispatch as routing_arithmetic
PROVIDERS.update({op: routing_arithmetic for op in ("AddFwdOp", "MulFwdOp", "NegFwdOp", "SubFwdOp", "LerpScalarFwdOp", "SumFwdOp", "MeanFwdOp")})

from .boolean_reduce import boolean_reduce
PROVIDERS.update({op: boolean_reduce for op in ("AllFwdOp", "AnyFwdOp")})

from .composed_activations import composed_activation
PROVIDERS.update({op: composed_activation for op in ('HardsigmoidFwdOp', 'HardswishFwdOp', 'LeakyReluFwdOp', 'PreluFwdOp', 'LerpTensorFwdOp')})

from .fused_silu import fused_silu
PROVIDERS["SiluAndMulFwdOp"] = fused_silu

# Full original compositions under explicit metadata dispatch; no case filtering.
def _varlen_with_native_rope(case):
    if case.params.get('pos_encoding_mode') == 'rope':
        from .varlen_rope import varlen_rope
        return varlen_rope(case)
    return segmented_attention(case)

def _prefill_with_native_rope(case):
    if case.params.get('fuse_rope'):
        from .prefill_rope import prefill_rope
        return prefill_rope(case)
    return prefill_append(case)

def _grouped_with_native_gate(case):
    if case.params.get('activation') == 'silu_and_mul':
        from .moe_gated import grouped_gated
        return grouped_gated(case)
    return grouped_gemm(case)

PROVIDERS.update({'GQAVarlenFwdOp': _varlen_with_native_rope,
                  'GQAPrefillPagedWithKVCacheFwdOp': _prefill_with_native_rope,
                  'MoEGroupedGemmFwdOp': _grouped_with_native_gate})

# Publish serially only after Chunk16 provider checks finish and parent review.
from .integration700_gdn_recurrent_registration1 import wrap as _integration700_gdn_recurrent_registration1_wrap
PROVIDERS["GDNFwdOp"] = _integration700_gdn_recurrent_registration1_wrap(PROVIDERS.get("GDNFwdOp"))

# Frozen original whole ChunkGatedDeltaRule exact16; retain all prior routes.
from .integration700_gdn_chunk_registration16 import wrap as _wrap_gdn_chunk16
PROVIDERS["GDNFwdOp"] = _wrap_gdn_chunk16(PROVIDERS.get("GDNFwdOp"))

# Parent publication only: frozen five original FlashAttn exactkeys.
from .integration700_attention_registration5 import wrap as _wrap_attention_registration5
for _attention_op in ('GQAPagedFwdOp', 'GQAVarlenFwdOp'):
    PROVIDERS[_attention_op] = _wrap_attention_registration5(PROVIDERS.get(_attention_op))

# Parent-only publication under ops_transformer publication lock.
from .integration700_causal_registration2 import wrap as _wrap_causal_registration2
PROVIDERS['CausalConv1dUpdateFwdOp'] = _wrap_causal_registration2(PROVIDERS.get('CausalConv1dUpdateFwdOp'))


from .integration700_gdn_local_registration10 import wrap as _wrap_gdn_local10
PROVIDERS['GDNFwdOp'] = _wrap_gdn_local10(PROVIDERS.get('GDNFwdOp'))

from .integration700_bf16_gemm_registration18 import wrap as _wrap_bf16_gemm18
PROVIDERS['GemmFwdOp'] = _wrap_bf16_gemm18(PROVIDERS.get('GemmFwdOp'))

from .integration700_bf16_bmm_registration7 import wrap as _wrap_bf16_bmm7
PROVIDERS['BmmFwdOp'] = _wrap_bf16_bmm7(PROVIDERS.get('BmmFwdOp'))
PROVIDERS['GroupedGemmFwdOp'] = _wrap_bf16_bmm7(PROVIDERS.get('GroupedGemmFwdOp'))

from .integration700_bf16_conv_registration1 import wrap as _wrap_bf16_conv1
PROVIDERS['Conv1dFwdOp'] = _wrap_bf16_conv1(PROVIDERS.get('Conv1dFwdOp'))

from .integration700_bf16_moe_masked_registration1 import wrap as _wrap_bf16_masked1
PROVIDERS['MoEGroupedGemmFwdOp'] = _wrap_bf16_masked1(PROVIDERS.get('MoEGroupedGemmFwdOp'))

from .integration700_fp16_bmm_registration1 import wrap as _wrap_fp16_bmm1
PROVIDERS['BmmFwdOp'] = _wrap_fp16_bmm1(PROVIDERS.get('BmmFwdOp'))

from .integration700_bf16_moe_down_registration3 import wrap as _wrap_bf16_moe_down3
PROVIDERS['MoEGroupedGemmFwdOp'] = _wrap_bf16_moe_down3(PROVIDERS.get('MoEGroupedGemmFwdOp'))

from .integration700_fp16_grouped_registration2 import wrap as _wrap_fp16_grouped2
PROVIDERS['GroupedGemmFwdOp'] = _wrap_fp16_grouped2(PROVIDERS.get('GroupedGemmFwdOp'))

from .integration700_bf16_grouped_registration1 import wrap as _wrap_bf16_grouped1
PROVIDERS['GroupedGemmFwdOp'] = _wrap_bf16_grouped1(PROVIDERS.get('GroupedGemmFwdOp'))

from ._integration700_fp8_m1 import wrap as _wrap_fp8_m1_700
PROVIDERS['GemmFP8FwdOp'] = _wrap_fp8_m1_700(PROVIDERS.get('GemmFP8FwdOp'))

from .integration700_threehost_native9 import install as _install_native9_700
_install_native9_700(PROVIDERS)

from .integration700_linear8 import install as _install_linear8_700
_install_linear8_700(PROVIDERS)

from .integration700_linear3 import install as _install_linear3_700
_install_linear3_700(PROVIDERS)

from .integration700_dsa_registration2 import install as _install_threehost_dsa_registration2_700
_install_threehost_dsa_registration2_700(PROVIDERS)

from .integration700_bmm_backend_variant import install as _install_threehost_bmm_variant_registration1_700
_install_threehost_bmm_variant_registration1_700(PROVIDERS)

from .integration700_moe_up import wrap as _moe_up_wrap
PROVIDERS['MoEGroupedGemmFwdOp'] = _moe_up_wrap(PROVIDERS.get('MoEGroupedGemmFwdOp'))

# Seven local FP8 cases verified by the single-agent experiment.
from .single_agent_local_fp8 import install as _install_single_local_fp8
_install_single_local_fp8(PROVIDERS)

# Single-agent server2: original NSA compressed attention, three exact cases.
from .single_server2_nsa_compressed import wrap as _single_server2_nsa_wrap
PROVIDERS['NSACompressedVarlenFwdOp'] = _single_server2_nsa_wrap(PROVIDERS['NSACompressedVarlenFwdOp'])
