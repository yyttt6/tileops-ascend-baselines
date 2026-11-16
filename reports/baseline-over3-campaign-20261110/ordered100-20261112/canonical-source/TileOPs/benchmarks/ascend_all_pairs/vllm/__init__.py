"""CPU-safe registration of concrete vllm-ascend candidates.

Importing PROVIDERS does not import torch, Triton, torch_npu, ACL or vendor code.
Factories require an external authorized NPU reservation in a future session.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from benchmarks.api import Case
    from benchmarks.ascend_baselines import Candidate

from .activation import activation
from .attention import packed_attention, mla_decode, dsa_decode, merge_states, lse_attention, nsa_selected, paged_append
from .basic import (fused_qk_norm_rope, adaptive_layer_norm, arithmetic, cumsum, gemm, group_norm,
                    grouped_gemm, mean, mean_pooling, normalization, pointwise_conv)
from .linear_attention import linear_attention
from .moe import expert
from .native import amax, causal_conv, dense_attention, fused_topk, mhc_post, softmax
from .rope import rope
from .cache import cache_transfer
from .reductions import average_pool, sum_reduction

PROVIDERS: dict[str, Callable[[Case], Candidate]] = {
    "AddFwdOp": arithmetic,
    "SubFwdOp": arithmetic,
    "NegFwdOp": arithmetic,
    "SigmoidFwdOp": activation,
    "SoftplusFwdOp": activation,
    "SiluFwdOp": activation,
    "SiluAndMulFwdOp": activation,
    "MeanFwdOp": mean,
    "SumFwdOp": sum_reduction,
    "AvgPool1dFwdOp": average_pool,
    "AvgPool2dFwdOp": average_pool,
    "AvgPool3dFwdOp": average_pool,
    "AdaptiveAvgPool2dFwdOp": average_pool,
    "MeanPoolingFwdOp": mean_pooling,
    "CumsumFwdOp": cumsum,
    "AmaxFwdOp": amax,
    "SoftmaxFwdOp": softmax,
    "RMSNormFwdOp": normalization,
    "LayerNormFwdOp": normalization,
    "FusedAddRMSNormFwdOp": normalization,
    "FusedAddLayerNormFwdOp": normalization,
    "AdaLayerNormFwdOp": adaptive_layer_norm,
    "GroupNormFwdOp": group_norm,
    "InstanceNormFwdOp": group_norm,
    "GemmFwdOp": gemm,
    "BmmFwdOp": gemm,
    "GroupedGemmFwdOp": grouped_gemm,
    "Conv1dFwdOp": pointwise_conv,
    "Conv2dFwdOp": pointwise_conv,
    "Conv3dFwdOp": pointwise_conv,
    "MoEGroupedGemmFwdOp": expert,
    "MoEExpertMLPFwdOp": expert,
    "SharedExpertMLPFwdOp": expert,
    "FusedTopKFwdOp": fused_topk,
    "RoPEFwdOp": rope,
    "RoPELlama31FwdOp": rope,
    "RoPENeoxPositionIdsFwdOp": rope,
    "LongRoPEFwdOp": rope,
    "YaRNFwdOp": rope,
    "DeltaNetFwdOp": linear_attention,
    "DeltaNetRecurrentFwdOp": linear_attention,
    "GDNFwdOp": linear_attention,
    "KDAFwdOp": linear_attention,
    "GQAPrefillPagedWithKVCacheFwdOp": paged_append,
    "FusedQKNormRoPEFwdOp": fused_qk_norm_rope,
    "MLAVarlenFwdOp": lse_attention,
    "NSACompressedVarlenFwdOp": lse_attention,
    "NSAVarlenFwdOp": nsa_selected,
    "PagedKVCacheGatherFwdOp": cache_transfer,
    "PagedKVCacheWriteFwdOp": cache_transfer,
    "MLAKVCacheWriteFwdOp": cache_transfer,
    "MergeAttentionStatesFwdOp": merge_states,
    "GQADenseFwdOp": dense_attention,
    "GQAVarlenFwdOp": packed_attention,
    "GQAPagedFwdOp": packed_attention,
    "MHADecodePagedWithKVCacheFwdOp": packed_attention,
    "MLADecodeWithKVCacheFwdOp": mla_decode,
    "DSADecodeWithKVCacheFwdOp": dsa_decode,
    "MHCPostFwdOp": mhc_post,
    "CausalConv1dFwdOp": causal_conv,
    "CausalConv1dUpdateFwdOp": causal_conv,
}

from .single_server1_fp64 import install as _install_single_server1_fp64
_install_single_server1_fp64(PROVIDERS)
