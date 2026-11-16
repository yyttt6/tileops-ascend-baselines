"""Manifest cases and named providers for the Ascend classic sample.

The shared adapters live beside NVIDIA's shared baseline resolvers; this bench
file owns the per-op implementation map. Use scripts/ascend_classic_check.py under
npu-guard: the upstream pytest Runner currently uses CUDA timing.
"""

from functools import partial

from benchmarks.ascend_transformer_ssd import transformer_ssd
from benchmarks.ascend_transformer_add import transformer_add

from benchmarks.ascend_baselines import (
    deepgemm_gemm, deepselect_amax, opsfft_fft, sgl_rms,
    tilekernels_add, vllm_add, vllm_rms,
)
from benchmarks.ascend_catlass_baselines import catlass_conv
from benchmarks.ascend_catlass_attention import catlass_gqa
from benchmarks.ascend_extra_baselines import atvc_add, fla_deltanet, sgl_deltanet, sgl_gqa
from benchmarks.ascend_math_baselines import opsmath_add
from benchmarks.ascend_tla_baselines import tla_add, tla_rope
from benchmarks.ascend_deepep_baselines import deepep_prepermute
from benchmarks.ascend_transform_baselines import tilekernels_fp8
from benchmarks.ascend_selection_baselines import tilekernels_amax
from benchmarks.ascend_tensor_baselines import opstensor_gemm
from benchmarks.ascend_tilelang_more import tilelang_add, tilelang_fp8, tilelang_gqa
from benchmarks.ascend_sgl_more import sgl_amax
from benchmarks.ascend_sgl_gemm import sgl_gemm
from benchmarks.ascend_vllm_more import vllm_gemm
from benchmarks.ascend_tla_reduction import tla_amax
from benchmarks.ascend_tla_attention import tla_gqa
from benchmarks.ascend_math_graph import opsmath_graph_amax
from benchmarks.ascend_nn_graph import (
    opsnn_graph_amax as opsnn_amax,
    opsnn_graph_conv as opsnn_conv, opsnn_graph_fp8 as opsnn_fp8,
    opsnn_graph_gemm as opsnn_gemm,
)
from benchmarks.ascend_nn_direct import opsnn_inplace_add as opsnn_add
from benchmarks.ascend_tensor_add import opstensor_add
from benchmarks.ascend_tensor_conv import opstensor_conv
from benchmarks.ascend_vllm_attention import vllm_gqa
from benchmarks.ascend_tilelang_generic import tilelang_generic_conv, tilelang_generic_ssd
from benchmarks.ascend_transformer_graph import (
    transformer_graph_fp8 as transformer_fp8,
    transformer_graph_gemm as transformer_gemm, transformer_graph_gqa as transformer_gqa,
    transformer_graph_prepermute as transformer_prepermute,
)
from benchmarks.ascend_tilelang_amax import tilelang_amax
from benchmarks.ascend_vllm_amax import vllm_amax
from benchmarks.ascend_vllm_delta import vllm_deltanet
from benchmarks.ascend_transformer_graph import transformer_graph_amax as transformer_amax
from benchmarks.ascend_tla_final import tla_deltanet, tla_prepermute, tla_rms
from benchmarks.ascend_flashmla_baselines import flashmla_amax, flashmla_gqa
from benchmarks.ascend_pointwise_conv import (
    tla_pointwise_conv, transformer_pointwise_conv, vllm_pointwise_conv,
)

from benchmarks.ascend_add_cases import sgl_add_cases as sgl_add
from benchmarks.ascend_gemm_cases import (
    tilelang_gemm_cases as tilelang_gemm, tla_gemm_cases as tla_gemm,
    catlass_gemm_cases as catlass_gemm, deepgemm_pointwise_conv_cases as deepgemm_pointwise_conv,
)
from benchmarks.ascend_rms_cases import (
    tilekernels_rms_cases as tilekernels_rms, tilelang_rms_cases as tilelang_rms,
    opsnn_rms_cases as opsnn_rms, transformer_rms_cases as transformer_rms,
)

from benchmarks.ascend_rope_cases import (
    tilekernels_rope_cases as tilekernels_rope, sgl_rope_cases as sgl_rope,
    vllm_rope_cases as vllm_rope, transformer_rope_cases as transformer_rope,
)

from benchmarks.ascend_quant_cases import opsnn_fp8_cases as opsnn_fp8, tilelang_fp8_cases as tilelang_fp8
from benchmarks.ascend_reduction_cases import (
    tilekernels_amax_cases as tilekernels_amax, sgl_amax_cases as sgl_amax,
    tla_amax_cases as tla_amax, opsmath_amax_cases as opsmath_graph_amax,
    transformer_amax_cases as transformer_amax, vllm_amax_cases as vllm_amax,
)


from benchmarks.ascend_attention_cases import (
    sgl_gqa_cases as sgl_gqa, vllm_gqa_cases as vllm_gqa,
    transformer_gqa_cases as transformer_gqa,
)
from benchmarks.ascend_catlass_attention_cases import catlass_gqa_cases as catlass_gqa
from benchmarks.ascend_tilelang_attention_cases import tilelang_gqa_cases as tilelang_gqa
from benchmarks.ascend_tensor_gemm_cases import opstensor_gemm_cases as opstensor_gemm
from benchmarks.ascend_tensor_conv_cases import opstensor_conv_cases as opstensor_conv
from benchmarks.ascend_catlass_conv_cases import catlass_conv_cases as catlass_conv
from benchmarks.ascend_tilelang_conv_cases import tilelang_conv_cases as tilelang_generic_conv
from benchmarks.ascend_sgl_large_rms import sgl_large_rms_cases as sgl_rms
from benchmarks.ascend_sgl_delta_cases import sgl_delta_cases as sgl_deltanet


from benchmarks.ascend_rope_attention_cases import (
    sgl_rope_gqa_cases as sgl_gqa, vllm_rope_gqa_cases as vllm_gqa,
)
from benchmarks.ascend_transformer_fp8_attention import transformer_fp8_gqa_cases as transformer_gqa
from benchmarks.ascend_catlass_bias_conv import catlass_conv_with_bias as catlass_conv
from benchmarks.ascend_fla_depthwise_conv import fla_depthwise_conv
from benchmarks.ascend_sgl_gemv_cases import sgl_gemv_cases as sgl_gemm
from benchmarks.ascend_tla_unweighted_rms import tla_unweighted_rms_cases
from benchmarks.ascend_tla_weighted_rms_rope import tla_weighted_rms_rope
from benchmarks.ascend_tla_packed_gqa import tla_packed_gqa as tla_gqa


def tla_rms(case):
    if case.inputs[1] is None:
        return tla_unweighted_rms_cases(case)
    return tla_weighted_rms_rope(case)


from benchmarks.ascend_bias_coordinate_conv import (
    catlass_bias_coordinate_cases as catlass_conv,
    tensor_bias_coordinate_cases as opstensor_conv,
    tla_bias_coordinate_cases as tla_pointwise_conv,
    tilelang_bias_coordinate_cases as tilelang_generic_conv,
    sgl_bias_coordinate_cases,
)

from benchmarks.ascend_tensor_aligned_gemm import (
    opstensor_aligned_gemm as opstensor_gemm,
    opstensor_aligned_conv_cases as opstensor_conv,
)

from benchmarks.ascend_catlass_l0bounded_conv import catlass_l0bounded_cases as catlass_conv

from benchmarks.ascend_tla_rope_simple import tla_simple_rope as tla_rope
from benchmarks.ascend_nn_aligned_gemm import opsnn_aligned_gemm as opsnn_gemm

def transformer_deltanet(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('Local direct ChunkKdaFwd currently dispatches a CANN builtin device kernel; own implementation attribution failed in classics_024/033. Local graph entry remains blocked by missing dtype inference. Do not credit builtin execution to this baseline.')


from benchmarks.ascend_tla_opt_delta import tla_opt_deltanet as tla_deltanet


from benchmarks.ascend_sgl_fp8_attention import sgl_fp8_gqa_cases as sgl_gqa


from benchmarks.ascend_vllm_fp8_attention import vllm_fp8_gqa_cases as vllm_gqa


PROVIDERS = {
    "AddFwdOp": {
        "ops-transformer": transformer_add,
        "TileKernels": tilekernels_add,
        "sgl-kernel-npu": sgl_add,
        "vllm-ascend": vllm_add,
        "atvc": atvc_add,
        "ops-math": opsmath_add,
        "tilelang-ascend": tla_add,
        "tilelang": tilelang_add,
        "ops-nn": opsnn_add,
        "ops-tensor": opstensor_add,
    },
    "RMSNormFwdOp": {
        "TileKernels": tilekernels_rms,
        "sgl-kernel-npu": sgl_rms,
        "vllm-ascend": vllm_rms,
        "tilelang": tilelang_rms,
        "ops-nn": opsnn_rms,
        "ops-transformer": transformer_rms,
        "tilelang-ascend": tla_rms,
    },
    "GemmFwdOp": {
        "DeepGEMM-Ascend": deepgemm_gemm,
        "catlass": catlass_gemm,
        "tilelang": tilelang_gemm,
        "tilelang-ascend": tla_gemm,
        "ops-tensor": opstensor_gemm,
        "sgl-kernel-npu": sgl_gemm,
        "vllm-ascend": vllm_gemm,
        "ops-nn": opsnn_gemm,
        "ops-transformer": transformer_gemm,
    },
    "Conv2dFwdOp": {"catlass": catlass_conv, "ops-nn": opsnn_conv,
                   "flash-linear-attention-npu": fla_depthwise_conv, "sgl-kernel-npu": sgl_bias_coordinate_cases,
                   "ops-tensor": opstensor_conv, "tilelang": tilelang_generic_conv,
                   "DeepGEMM-Ascend": deepgemm_pointwise_conv, "tilelang-ascend": tla_pointwise_conv,
                   "ops-transformer": transformer_pointwise_conv, "vllm-ascend": vllm_pointwise_conv},
    "RoPEFwdOp": {
        "tilelang-ascend": tla_rope,
        "TileKernels": tilekernels_rope,
        "ops-transformer": transformer_rope,
        "sgl-kernel-npu": sgl_rope,
        "vllm-ascend": vllm_rope,
    },
    "FP8QuantFwdOp": {"TileKernels": tilekernels_fp8, "tilelang": tilelang_fp8, "ops-nn": opsnn_fp8,
                     "ops-transformer": transformer_fp8},
    "MoEPrePermuteFwdOp": {"DeepEP-Ascend": deepep_prepermute, "ops-transformer": transformer_prepermute,
                         "tilelang-ascend": tla_prepermute},
    "GQADenseFwdOp": {"sgl-kernel-npu": sgl_gqa, "catlass": catlass_gqa, "tilelang": tilelang_gqa,
                     "tilelang-ascend": tla_gqa, "vllm-ascend": vllm_gqa, "ops-transformer": transformer_gqa,
                     "FlashMLA": flashmla_gqa},
    "DeltaNetFwdOp": {
        "sgl-kernel-npu": sgl_deltanet,
        "flash-linear-attention-npu": fla_deltanet,
        "ops-transformer": transformer_deltanet,
        "tilelang-ascend": tla_deltanet,
        "vllm-ascend": vllm_deltanet,
    },
    "AmaxFwdOp": {"DeepSelect": deepselect_amax, "TileKernels": tilekernels_amax, "sgl-kernel-npu": sgl_amax,
                 "tilelang-ascend": tla_amax, "ops-math": opsmath_graph_amax, "ops-nn": opsnn_amax,
                 "tilelang": tilelang_amax, "vllm-ascend": vllm_amax,
                 "ops-transformer": transformer_amax, "FlashMLA": flashmla_amax},
    "FFTC2CFwdOp": {"ops-fft": opsfft_fft},
    "SSDChunkStateFwdOp": {"tilelang": tilelang_generic_ssd, "ops-transformer": transformer_ssd},
}


def candidate_factories(case):
    """Resolve each provider separately so a failed import cannot hide its peers."""
    return {library: partial(factory, case)
            for library, factory in PROVIDERS.get(case._call.signature.name, {}).items()}
