"""Original flash-linear-attention-npu providers; importing this is CPU-only."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from benchmarks.api import Case
    from benchmarks.ascend_baselines import Candidate


def cumsum(case):
    from .providers import cumsum as build
    return build(case)


def reduction(case):
    from .providers import reduce
    return reduce(case)


def pointwise(case):
    from .providers import pointwise as build
    return build(case)


def recurrent(case):
    from .providers import recurrent as build
    return build(case)


def gdn(case):
    import os
    from .providers import gdn as build, recurrent, chunk_kda
    if os.environ.get("FLA_ALLPAIRS_MODE") == "chunk_kda":return chunk_kda(case)
    automatic=os.environ.get("FLA_ALLPAIRS_MODE") is None and str(case.inputs[0].dtype)=="torch.bfloat16" and (case.inputs[0].shape[1]<=8 or case.params.get("use_gate_in_kernel"))
    return recurrent(case) if automatic or os.environ.get("FLA_ALLPAIRS_MODE") in ("recurrent","recurrent_vfirst") else build(case)


def kda(case):
    import os
    from .providers import recurrent,chunk_kda
    return chunk_kda(case) if os.environ.get("FLA_ALLPAIRS_MODE")=="chunk_kda" else recurrent(case)


def deltanet(case):
    import os
    if os.environ.get("FLA_ALLPAIRS_MODE") == "recurrent" or (os.environ.get("FLA_ALLPAIRS_MODE") is None and str(case.inputs[0].dtype)=="torch.bfloat16" and case.inputs[0].shape[1]<=8):
        from .providers import recurrent
        return recurrent(case)
    from .providers import select_deployment
    select_deployment("aclnnChunkGatedDeltaRuleFwd")
    from benchmarks.ascend_extra_baselines import fla_deltanet
    return fla_deltanet(case)


def convolution(case):
    from .spatial import convolution as build
    return build(case)


def average_pool(case):
    from .spatial import average_pool as build
    return build(case)


def rmsnorm(case):
    from .providers import rmsnorm as build
    return build(case)


def adaptive_average_pool(case):
    from .spatial import adaptive_average_pool as build
    return build(case)


def mean_pooling(case):
    from .spatial import mean_pooling as build
    return build(case)


def causal(case):
    from .providers import causal as build
    return build(case)


def bmm(case):
    import os
    if str(case.inputs[0].dtype)=='torch.bfloat16':
        from .cube_matmul import gemm as build
        return build(case)
    from .providers import bmm as build
    return build(case)


def gemm(case):
    import os
    if str(case.inputs[0].dtype)=='torch.bfloat16':
        from .cube_matmul import gemm as build
        return build(case)
    from .spatial import gemm as build
    return build(case)


def centered_norm(case):
    from .normalization import centered
    return centered(case)


def lerp(case):
    from .normalization import lerp as build
    return build(case)


def variance(case):
    from .variance import variance as build
    return build(case)


def gla(case):
    from .gla import gla as build
    return build(case)


def exponential(case):
    from .exponential import exponential as build
    return build(case)


def batchnorm(case):
    from .batchnorm import batchnorm as build
    return build(case)


def vectornorm(case):
    from .vectornorm import vectornorm as build
    return build(case)


def sigmoid(case):
    import os
    if os.environ.get('FLA_ALLPAIRS_MODE')=='partial_native':
        from .gate_activations import gate_activation
        return gate_activation(case)
    from .sigmoid import sigmoid as build
    return build(case)


def absolute(case):
    from .absolute import absolute as build
    return build(case)


def deviation(case):
    from .deviation import deviation as build
    return build(case)


def dequant(case):
    from .dequant import dequant as build
    return build(case)


def hadamard(case):
    from .hadamard import hadamard as build
    return build(case)


def fused_qk(case):
    from .fused_qk import fused_qk as build
    return build(case)


def grouped_gemm(case):
    from .grouped_gemm import grouped_gemm as build
    return build(case)


def fp8_matmul(case):
    from .fp8_matmul import fp8_matmul as build
    return build(case)


def int8_matmul(case):
    from .int8_matmul import int8_matmul as build
    return build(case)


def gate_activation(case):
    from .gate_activations import gate_activation as build
    return build(case)


def delta_backward(case):
    from .backward import deltanet_chunk_backward as build
    return build(case)


def delta_chunk_forward(case):
    from .chunk_forward import delta_chunk_forward as build
    return build(case)


def piecewise(case):
    from .piecewise import piecewise as build
    return build(case)


def mhc_post(case):
    from .sequence import mhc_post as build
    return build(case)


def state_passing(case):
    from .sequence import state_passing as build
    return build(case)


def chunk_coupling(case):
    from .sequence import chunk_coupling as build
    return build(case)


def chunk_cumsum(case):
    from .sequence import chunk_cumsum as build
    return build(case)


def selection(case):
    from .selection import selection as build
    return build(case)


PROVIDERS: dict[str, Callable[[Case], Candidate]] = {"GemmFP8FwdOp":fp8_matmul,"BmmFP8FwdOp":fp8_matmul,"INT8DequantPerTensorFwdOp":dequant,"INT8DequantPerChannelFwdOp":dequant,"INT8DequantPerBlockFwdOp":dequant,"HadamardTransformFwdOp":hadamard,"FusedQKNormRoPEFwdOp":fused_qk,"GroupedGemmFwdOp":grouped_gemm,"AbsFwdOp":absolute,"StdFwdOp":deviation,"SigmoidFwdOp":sigmoid,"TanhFwdOp":sigmoid,"VectorNormFwdOp":vectornorm,"BatchNormFwdOp":batchnorm,"BatchNormBwdOp":batchnorm,"CumsumFwdOp": cumsum, "SumFwdOp": reduction, "MeanFwdOp": reduction,
             "GemmINT8W8A8FwdOp":int8_matmul,
             "MHCPostFwdOp":mhc_post,"SSDStatePassingFwdOp":state_passing,"SSDChunkCouplingFwdOp":chunk_coupling,
             **dict.fromkeys(('ReluFwdOp','LeakyReluFwdOp','PreluFwdOp','HardtanhFwdOp','HardsigmoidFwdOp','HardswishFwdOp','ClampScalarFwdOp','ClampTensorFwdOp','MaximumFwdOp','MinimumFwdOp'),piecewise),
             **dict.fromkeys(('WhereFwdOp','MaskedFillScalarFwdOp','MaskedFillTensorFwdOp'),selection),
             **dict.fromkeys(("AddFwdOp", "SubFwdOp", "MulFwdOp", "NegFwdOp", "SiluFwdOp", "SiluAndMulFwdOp"), pointwise),
             "DeltaNetFwdOp": deltanet, "DeltaNetRecurrentFwdOp": recurrent,
             "KDAFwdOp": kda, "GDNFwdOp": gdn,
             **dict.fromkeys(("LayerNormFwdOp","FusedAddLayerNormFwdOp","AdaLayerNormFwdOp","AdaLayerNormZeroFwdOp","GroupNormFwdOp","InstanceNormFwdOp"),centered_norm),
             "RMSNormFwdOp":rmsnorm,"FusedAddRMSNormFwdOp":rmsnorm,
             "AdaptiveAvgPool2dFwdOp":adaptive_average_pool,"MeanPoolingFwdOp":mean_pooling,
             "CausalConv1dFwdOp":causal,"CausalConv1dUpdateFwdOp":causal,
             "BmmFwdOp":bmm,
             "GemmFwdOp":gemm,
             "ExpFwdOp":exponential,"Expm1FwdOp":exponential,
             "GLAFwdOp":gla,"GLARecurrentFwdOp":gla,"GLAChunkFwdOp":gla,
             "VarFwdOp":variance,"VarMeanFwdOp":variance,
             "LerpScalarFwdOp":lerp,"LerpTensorFwdOp":lerp,
             **dict.fromkeys(("Conv1dFwdOp","Conv2dFwdOp","Conv3dFwdOp"),convolution),
             **dict.fromkeys(("AvgPool1dFwdOp","AvgPool2dFwdOp","AvgPool3dFwdOp"),average_pool)}

from pathlib import Path as _Path
if (_Path(__file__).resolve().parents[4]/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla/partial_native_ready.json').is_file():
    PROVIDERS['SoftplusFwdOp']=gate_activation
    PROVIDERS['DeltaNetChunkBwdOp']=delta_backward
    PROVIDERS['SSDChunkCumsumFwdOp']=chunk_cumsum
    PROVIDERS['DeltaNetChunkFwdOp']=delta_chunk_forward


def _include_host_work(factory):
    # Public import remains CPU-only. All data-dependent preparation stays in
    # run; wall-clock timing includes Python packing and metadata round trips.
    def build(case):
        from dataclasses import replace
        return replace(factory(case),timing_mode="host_roundtrip")
    build.__name__=factory.__name__
    build.__doc__=factory.__doc__
    return build


PROVIDERS={name:_include_host_work(factory) for name,factory in PROVIDERS.items()}

# Exact-key registration baseline700/fla 131331
def _integration700_fla_route(previous, exact_ids):
    def factory(case):
        if case.id in exact_ids:
            from .integration700_fla_native import build
            return build(case)
        return previous(case)
    return factory
PROVIDERS['AddFwdOp'] = _integration700_fla_route(PROVIDERS['AddFwdOp'], ['hidden-state-prefill-float16'])
PROVIDERS['CumsumFwdOp'] = _integration700_fla_route(PROVIDERS['CumsumFwdOp'], ['hidden-state-float16'])
PROVIDERS['GemmFwdOp'] = _integration700_fla_route(PROVIDERS['GemmFwdOp'], ['ds-v3-m2-qkv-a-bfloat16'])

# Exact-key registration baseline700/fla 131753
def _integration700_fla_route(previous, exact_ids):
    def factory(case):
        if case.id in exact_ids:
            from .integration700_fla_native import build
            return build(case)
        return previous(case)
    return factory
PROVIDERS['AddFwdOp'] = _integration700_fla_route(PROVIDERS['AddFwdOp'], ['hidden-state-prefill-float16', 'linear-bias-float32'])
PROVIDERS['CumsumFwdOp'] = _integration700_fla_route(PROVIDERS['CumsumFwdOp'], ['hidden-state-float16'])
PROVIDERS['GDNFwdOp'] = _integration700_fla_route(PROVIDERS['GDNFwdOp'], ['q35-35b-4096-float16'])
PROVIDERS['GLAFwdOp'] = _integration700_fla_route(PROVIDERS['GLAFwdOp'], ['decode-first-token-float16'])
PROVIDERS['GemmFwdOp'] = _integration700_fla_route(PROVIDERS['GemmFwdOp'], ['ds-v3-m2-qkv-a-bfloat16', 'ds-v3-m64-down-bfloat16'])

# Exact-key registration baseline700/fla 131906
def _integration700_fla_route(previous, exact_ids):
    def factory(case):
        if case.id in exact_ids:
            from .integration700_fla_native import build
            return build(case)
        return previous(case)
    return factory
PROVIDERS['AddFwdOp'] = _integration700_fla_route(PROVIDERS['AddFwdOp'], ['hidden-state-prefill-float16', 'linear-bias-float32'])
PROVIDERS['CumsumFwdOp'] = _integration700_fla_route(PROVIDERS['CumsumFwdOp'], ['hidden-state-float16'])
PROVIDERS['GDNFwdOp'] = _integration700_fla_route(PROVIDERS['GDNFwdOp'], ['q35-35b-4096-float16'])
PROVIDERS['GLAFwdOp'] = _integration700_fla_route(PROVIDERS['GLAFwdOp'], ['decode-first-token-float16'])
PROVIDERS['GemmFwdOp'] = _integration700_fla_route(PROVIDERS['GemmFwdOp'], ['ds-v3-m2-qkv-a-bfloat16', 'ds-v3-m64-down-bfloat16'])
PROVIDERS['GroupedGemmFwdOp'] = _integration700_fla_route(PROVIDERS['GroupedGemmFwdOp'], ['moe-fwd-bfloat16', 'qwen3-30b-ep8-empty-bfloat16'])

# Exact-key registration baseline700/fla 132211
def _integration700_fla_route(previous, exact_ids):
    def factory(case):
        if case.id in exact_ids:
            from .integration700_fla_native import build
            return build(case)
        return previous(case)
    return factory
PROVIDERS['AddFwdOp'] = _integration700_fla_route(PROVIDERS['AddFwdOp'], ['hidden-state-prefill-float16', 'linear-bias-float32'])
PROVIDERS['CumsumFwdOp'] = _integration700_fla_route(PROVIDERS['CumsumFwdOp'], ['hidden-state-float16'])
PROVIDERS['GDNFwdOp'] = _integration700_fla_route(PROVIDERS['GDNFwdOp'], ['prefill-12k-float16', 'q35-35b-4096-float16', 'q35-35b-8192-float16', 'q35-397b-tp1-2048-float16', 'q35-397b-tp1-4096-float16'])
PROVIDERS['GLAFwdOp'] = _integration700_fla_route(PROVIDERS['GLAFwdOp'], ['decode-first-token-float16'])
PROVIDERS['GemmFwdOp'] = _integration700_fla_route(PROVIDERS['GemmFwdOp'], ['ds-v3-m2-qkv-a-bfloat16', 'ds-v3-m64-down-bfloat16'])
PROVIDERS['GroupedGemmFwdOp'] = _integration700_fla_route(PROVIDERS['GroupedGemmFwdOp'], ['moe-fwd-bfloat16', 'qwen3-30b-ep8-empty-bfloat16'])

# Exact-key registration baseline700/fla 132433
def _integration700_fla_route(previous, exact_ids):
    def factory(case):
        if case.id in exact_ids:
            from .integration700_fla_native import build
            return build(case)
        return previous(case)
    return factory
PROVIDERS['GDNFwdOp'] = _integration700_fla_route(PROVIDERS['GDNFwdOp'], ['q35-35b-32768-float16', 'q35-397b-tp1-16384-float16', 'q35-397b-tp1-32768-bfloat16', 'q35-397b-tp1-8192-float16'])

# Exact-key registration baseline700/fla 133408
def _integration700_fla_route(previous, exact_ids):
    def factory(case):
        if case.id in exact_ids:
            from .integration700_fla_native import build
            return build(case)
        return previous(case)
    return factory
PROVIDERS['GDNFwdOp'] = _integration700_fla_route(PROVIDERS['GDNFwdOp'], ['q35-35b-65536-float16', 'q35-397b-tp1-32768-float16', 'q35-397b-tp1-65536-float16', 'q35-397b-tp2-32768-float16', 'q35-397b-tp8-16384-float16'])

# Exact-key registration baseline700/fla 134058
def _integration700_fla_route(previous, exact_ids):
    def factory(case):
        if case.id in exact_ids:
            from .integration700_fla_native import build
            return build(case)
        return previous(case)
    return factory
PROVIDERS['GDNFwdOp'] = _integration700_fla_route(PROVIDERS['GDNFwdOp'], ['q35-27b-65536-bfloat16', 'q35-35b-8x8192-float16', 'q35-397b-tp1-8x8192-float16'])

# Exact-key registration baseline700/fla 134407
def _integration700_fla_route(previous, exact_ids):
    def factory(case):
        if case.id in exact_ids:
            from .integration700_fla_native import build
            return build(case)
        return previous(case)
    return factory
PROVIDERS['GDNFwdOp'] = _integration700_fla_route(PROVIDERS['GDNFwdOp'], ['q35-35b-16x8192-bfloat16', 'q35-397b-tp1-8x8192-bfloat16', 'q35-397b-tp2-16x8192-bfloat16'])

# Exact-key registration baseline700/fla 134732
def _integration700_fla_route(previous, exact_ids):
    def factory(case):
        if case.id in exact_ids:
            from .integration700_fla_native import build
            return build(case)
        return previous(case)
    return factory
PROVIDERS['GDNFwdOp'] = _integration700_fla_route(PROVIDERS['GDNFwdOp'], ['q35-2b-32x8192-bfloat16', 'q35-35b-32x8192-bfloat16'])

from .single_server1_fp64 import install as _install_single_server1_fp64
_install_single_server1_fp64(PROVIDERS)
