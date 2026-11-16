"""SGL's own implementations; importing this registry never imports torch/NPU."""
from .elementwise import sigmoid, silu, sub, neg, silu_and_mul
from .reduction import argmax, mean
from .norm import layernorm, fused_add_rms, groupnorm, instancenorm

PROVIDERS = {'SigmoidFwdOp': sigmoid, 'SiluFwdOp': silu, 'SubFwdOp': sub,
             'NegFwdOp': neg, 'SiluAndMulFwdOp': silu_and_mul, 'ArgmaxFwdOp': argmax,
             'MeanFwdOp': mean, 'LayerNormFwdOp': layernorm,
             'FusedAddRMSNormFwdOp': fused_add_rms, 'GroupNormFwdOp': groupnorm,
             'InstanceNormFwdOp': instancenorm}

from .matmul import bmm, grouped_gemm
from .cache import paged_cache_write
from .linear import gdn, kda, delta_recurrent, delta_chunk
from . import classics
PROVIDERS.update({'BmmFwdOp':bmm,'GroupedGemmFwdOp':grouped_gemm,
                  'PagedKVCacheWriteFwdOp':paged_cache_write,'GDNFwdOp':gdn,'KDAFwdOp':kda,
                  'DeltaNetRecurrentFwdOp':delta_recurrent,'DeltaNetChunkFwdOp':delta_chunk,
                  'AddFwdOp':classics.add,'AmaxFwdOp':classics.amax,'RMSNormFwdOp':classics.rms,
                  'GemmFwdOp':classics.gemm,'Conv2dFwdOp':classics.conv2d,'RoPEFwdOp':classics.rope,
                  'DeltaNetFwdOp':classics.delta,'GQADenseFwdOp':classics.gqa})
from .rope import rope_variant
from .attention import varlen, paged
PROVIDERS.update({op:rope_variant for op in ('RoPELlama31FwdOp','YaRNFwdOp','LongRoPEFwdOp','RoPENeoxPositionIdsFwdOp')})
PROVIDERS.update({'GQAVarlenFwdOp':varlen,'GQAPagedFwdOp':paged})
from .sampling import chain, categorical
PROVIDERS.update({'ChainSpeculativeSamplingFwdOp':chain,'SamplingFromProbsFwdOp':categorical})
from .additional import mul, maximum, fused_add_layernorm, ada_layernorm, mean_pooling
PROVIDERS.update({'MulFwdOp':mul,'MaximumFwdOp':maximum,'FusedAddLayerNormFwdOp':fused_add_layernorm,
                  'AdaLayerNormFwdOp':ada_layernorm,'MeanPoolingFwdOp':mean_pooling})
from .attention import mla_decode, mha_paged
PROVIDERS.update({'MLADecodeWithKVCacheFwdOp':mla_decode,'MHADecodePagedWithKVCacheFwdOp':mha_paged})
from .moe import grouped, expert_mlp
PROVIDERS.update({'MoEGroupedGemmFwdOp':grouped,'MoEExpertMLPFwdOp':expert_mlp})
from .additional import ada_zero, minimum
from .selection_reduce import argmin, amin
PROVIDERS.update({'AdaLayerNormZeroFwdOp':ada_zero,'MinimumFwdOp':minimum,'ArgminFwdOp':argmin,'AminFwdOp':amin})
from .attention import dsa_decode
PROVIDERS['DSADecodeWithKVCacheFwdOp']=dsa_decode

from .pooling import pool
PROVIDERS.update({op:pool for op in ('AvgPool1dFwdOp','AvgPool2dFwdOp','AvgPool3dFwdOp','MaxPool1dFwdOp','MaxPool1dIndicesFwdOp','MaxPool2dFwdOp','MaxPool2dIndicesFwdOp','MaxPool3dFwdOp','MaxPool3dIndicesFwdOp','AdaptiveAvgPool2dFwdOp','AdaptiveMaxPool2dFwdOp','AdaptiveMaxPool2dIndicesFwdOp')})

from .scalar import selectors, lerp_scalar, sum_reduce
PROVIDERS.update({op:selectors for op in ('AbsFwdOp','ReluFwdOp','ClampScalarFwdOp','ClampTensorFwdOp','HardtanhFwdOp')})
PROVIDERS.update({'LerpScalarFwdOp':lerp_scalar,'SumFwdOp':sum_reduce})
from .attention import prefill_append
PROVIDERS['GQAPrefillPagedWithKVCacheFwdOp']=prefill_append
from .convolution import conv_nd
PROVIDERS.update({'Conv1dFwdOp':conv_nd,'Conv3dFwdOp':conv_nd})
from .indexer import topk_select
PROVIDERS['TopKSelectFwdOp']=topk_select
from .attention import nsa_selected
PROVIDERS['NSAVarlenFwdOp']=nsa_selected
from .batchnorm import batchnorm
PROVIDERS['BatchNormFwdOp']=batchnorm
from .convolution import conv2d as full_window_conv2d
PROVIDERS['Conv2dFwdOp']=full_window_conv2d
from .norm_compat import fused_add_rms as compatible_fused_add_rms, instancenorm as complete_instancenorm
PROVIDERS.update({'FusedAddRMSNormFwdOp':compatible_fused_add_rms,'InstanceNormFwdOp':complete_instancenorm})
from .composed import activation as composed_activation, variance
PROVIDERS.update({op:composed_activation for op in ('HardsigmoidFwdOp','HardswishFwdOp','LeakyReluFwdOp','PreluFwdOp')})
PROVIDERS.update({'VarFwdOp':variance,'VarMeanFwdOp':variance})
from .mean_compat import mean as compatible_mean
PROVIDERS['MeanFwdOp']=compatible_mean
from .conditional import select
PROVIDERS.update({op:select for op in ('WhereFwdOp','MaskedFillScalarFwdOp','MaskedFillTensorFwdOp')})
from .extended import causal_conv,qk_norm_rope
PROVIDERS.update({'CausalConv1dFwdOp':causal_conv,'CausalConv1dUpdateFwdOp':causal_conv,'FusedQKNormRoPEFwdOp':qk_norm_rope})
from .mla_cache import write as mla_cache_write
PROVIDERS['MLAKVCacheWriteFwdOp']=mla_cache_write
from .topk_mask import build as topk_mask
PROVIDERS['TopKMaskFwdOp']=topk_mask
from .gather import build as paged_gather
PROVIDERS['PagedKVCacheGatherFwdOp']=paged_gather
from .softplus import build as softplus
PROVIDERS['SoftplusFwdOp']=softplus
from .extra_arithmetic import dequant, tanh, lerp
PROVIDERS.update({op:dequant for op in ('INT8DequantPerTensorFwdOp','INT8DequantPerChannelFwdOp','INT8DequantPerBlockFwdOp')})
PROVIDERS.update({'TanhFwdOp':tanh,'LerpTensorFwdOp':lerp})
from .chunk_complete import build as complete_delta_chunk
PROVIDERS['DeltaNetChunkFwdOp'] = complete_delta_chunk
from .extra_arithmetic import lerp_scalar as complete_lerp_scalar
PROVIDERS['LerpScalarFwdOp'] = complete_lerp_scalar
from .row_gemv import mean as complete_row_mean, sum_reduce as complete_row_sum
PROVIDERS.update({'MeanFwdOp':complete_row_mean,'SumFwdOp':complete_row_sum})
from .boolean_reduce import build as boolean_reduce
PROVIDERS.update({op:boolean_reduce for op in ('AnyFwdOp','AllFwdOp','CountNonzeroFwdOp')})
from .vector_norm import build as vector_norm
PROVIDERS['VectorNormFwdOp'] = vector_norm
from .boolean_elementwise import build as boolean_elementwise
PROVIDERS.update({op:boolean_elementwise for op in ('BitwiseAndFwdOp','BitwiseOrFwdOp','BitwiseXorFwdOp','LogicalAndFwdOp','LogicalOrFwdOp','LogicalNotFwdOp')})
from .quant_gemv import build as quant_gemv
PROVIDERS.update({op:quant_gemv for op in ('BmmFP8FwdOp','GemmFP8FwdOp','GemmINT8W8A8FwdOp')})
from .comparison import build as comparison
PROVIDERS.update({op:comparison for op in ('EqFwdOp','NeFwdOp','GeFwdOp','GtFwdOp','LeFwdOp','LtFwdOp')})
from .finite_truth import build as finite_truth
PROVIDERS.update({op:finite_truth for op in ('AnyFwdOp','AllFwdOp','CountNonzeroFwdOp','LogicalAndFwdOp','LogicalOrFwdOp','LogicalNotFwdOp')})
from .lightning_logits import build as lightning_logits
PROVIDERS['FP8LightningIndexerFwdOp'] = lightning_logits
from .exp_composed import build as exp_composed
PROVIDERS.update({op:exp_composed for op in ('ExpFwdOp','Expm1FwdOp','EluFwdOp','SeluFwdOp')})
from .sign import build as sign
PROVIDERS['SignFwdOp'] = sign
from .inverse_rms import build as positive_inverse
PROVIDERS.update({op:positive_inverse for op in ('ReciprocalFwdOp','DivFwdOp')})
from .std import build as std
PROVIDERS['StdFwdOp'] = std
from .mish import build as mish
PROVIDERS['MishFwdOp'] = mish
from .alibi import build as alibi
PROVIDERS['AlibiFwdOp'] = alibi
from .bitwise_not import build as bitwise_not
PROVIDERS['BitwiseNotFwdOp'] = bitwise_not
from .probability_composed import build as probability_composed
PROVIDERS.update({op:probability_composed for op in ('SoftmaxFwdOp','MinPMaskFwdOp')})
from .w4a16 import build as w4a16
PROVIDERS['GemmW4A16FwdOp'] = w4a16
from .moe_post import build as moe_post
PROVIDERS['MoEPostPermuteFwdOp'] = moe_post
from .gelu_tanh_mul import build as gelu_tanh_mul
PROVIDERS['GeluTanhAndMulFwdOp'] = gelu_tanh_mul
from .dropout import build as dropout
PROVIDERS['DropoutFwdOp'] = dropout
from .mhc_post import build as mhc_post
PROVIDERS['MHCPostFwdOp'] = mhc_post
from .extended_truth import build as extended_truth
PROVIDERS.update({op:extended_truth for op in ('AnyFwdOp','AllFwdOp','CountNonzeroFwdOp')})
from .ssd_components import cumsum as ssd_cumsum, coupling as ssd_coupling
PROVIDERS.update({'SSDChunkCumsumFwdOp':ssd_cumsum,'SSDChunkCouplingFwdOp':ssd_coupling})

# Exact keys privately verified by baseline700 sgl_conv_register9; preserve prior providers elsewhere.
# The frozen adapter is private to this integration task; load it without copying or mutating vendor code.
import importlib.util as _i700_sgl_conv_register9_importlib_util
from pathlib import Path as _i700_sgl_conv_register9_Path
_i700_sgl_conv_register9_file = _i700_sgl_conv_register9_Path(__file__).resolve().parents[4] / "reports/baseline-integration-700-20261026/sgl_conv_register9/integration700_sgl_conv_register9.py"
_i700_sgl_conv_register9_spec = _i700_sgl_conv_register9_importlib_util.spec_from_file_location("integration700_sgl_conv_register9", _i700_sgl_conv_register9_file)
_i700_sgl_conv_register9_module = _i700_sgl_conv_register9_importlib_util.module_from_spec(_i700_sgl_conv_register9_spec)
_i700_sgl_conv_register9_spec.loader.exec_module(_i700_sgl_conv_register9_module)
_i700_sgl_conv_register9_conv = _i700_sgl_conv_register9_module.conv
_i700_sgl_conv_register9_chain = _i700_sgl_conv_register9_module.chain

def _i700_sgl_conv_register9_dispatch(previous, factory, ids):
    def build(case):
        return factory(case) if case.id in ids else previous(case)
    return build

_i700_sgl_conv_register9_conv_ids = (
    "resnet-1x1-float16", "bottleneck-expand-float16", "late-stage-1x1-bias-float16",
    "classifier-1x1-float16", "resnet-1x1-bfloat16", "resnet-1x1-bias-bfloat16",
    "classifier-1x1-bias-float16",
)
_i700_sgl_conv_register9_chain_ids = ("llama2-7b-b256-n1", "llama-8b-b17-n8")
PROVIDERS["Conv2dFwdOp"] = _i700_sgl_conv_register9_dispatch(PROVIDERS["Conv2dFwdOp"], _i700_sgl_conv_register9_conv, _i700_sgl_conv_register9_conv_ids)
PROVIDERS["ChainSpeculativeSamplingFwdOp"] = _i700_sgl_conv_register9_dispatch(PROVIDERS["ChainSpeculativeSamplingFwdOp"], _i700_sgl_conv_register9_chain, _i700_sgl_conv_register9_chain_ids)

# sgl_dsa_source_reconcile1: canonical full-output pass, historical source race reconciled.
from .integration700_sgl_dsa_source_reconcile1 import build as _dsa_sr1_build, KEYS as _dsa_sr1_keys
def _dsa_sr1_wrap(previous):
    def route(case):
        if case.id in _dsa_sr1_keys:
            return _dsa_sr1_build(case)
        if previous is not None:
            return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No SGL provider for this unregistered exact key')
    return route
PROVIDERS['DSADecodeWithKVCacheFwdOp'] = _dsa_sr1_wrap(PROVIDERS.get('DSADecodeWithKVCacheFwdOp'))

from .single_server1_fp64 import install as _install_single_server1_fp64
_install_single_server1_fp64(PROVIDERS)
