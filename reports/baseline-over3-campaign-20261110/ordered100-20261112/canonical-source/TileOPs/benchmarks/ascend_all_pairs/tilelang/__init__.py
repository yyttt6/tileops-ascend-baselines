"""Isolated TileLang providers; no device runtime imports at module scope."""
from __future__ import annotations
from typing import Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from benchmarks.api import Case
    from benchmarks.ascend_baselines import Candidate

from .adapters import bmm, convolution, fused_add_rms, grouped_gemm, sum_mean, moe_grouped
from .attention import dense, paged
from .fp8 import fp8_scaled
from .microblocks import bitwise, clamp, small_arithmetic

def _existing(module, name):
    def factory(case):
        import importlib
        return getattr(importlib.import_module(module), name)(case)
    return factory

PROVIDERS: dict[str, Callable[[Case], Candidate]] = {
    "GemmFP8FwdOp": fp8_scaled,
    "BmmFP8FwdOp": fp8_scaled,
    "BitwiseNotFwdOp": bitwise,
    "BitwiseXorFwdOp": bitwise,
    "ClampScalarFwdOp": clamp,
    "ClampTensorFwdOp": clamp,
    "HardtanhFwdOp": clamp,
    "GQADenseFwdOp": dense,
    "GQAPagedFwdOp": paged,
    "SumFwdOp": sum_mean,
    "MeanFwdOp": sum_mean,
    "MoEGroupedGemmFwdOp": moe_grouped,
    "BmmFwdOp": bmm,
    "Conv1dFwdOp": convolution,
    "Conv2dFwdOp": convolution,
    "Conv3dFwdOp": convolution,
    "GroupedGemmFwdOp": grouped_gemm,
    "FusedAddRMSNormFwdOp": fused_add_rms,
    "AddFwdOp": _existing("benchmarks.ascend_tilelang_more", "tilelang_add"),
    "GemmFwdOp": _existing("benchmarks.ascend_gemm_cases", "tilelang_gemm_cases"),
    "RMSNormFwdOp": _existing("benchmarks.ascend_rms_cases", "tilelang_rms_cases"),
    "FP8QuantFwdOp": _existing("benchmarks.ascend_quant_cases", "tilelang_fp8_cases"),
}

PROVIDERS.update({op:clamp for op in ('ReluFwdOp','MaximumFwdOp','MinimumFwdOp')})
PROVIDERS.update({op:small_arithmetic for op in ('AbsFwdOp','DivFwdOp','ReciprocalFwdOp')})
PROVIDERS['LogicalNotFwdOp']=bitwise

from .reduce import extrema
PROVIDERS.update({op:extrema for op in ('AmaxFwdOp','AminFwdOp','AllFwdOp','AnyFwdOp')})

from .sum_reduction import sum_fwd
from .pool import fixed as fixed_pool, adaptive as adaptive_pool, mean_chunks
from .attention_crops import dense as cropped_dense, paged as cropped_paged
PROVIDERS['SumFwdOp']=sum_fwd
PROVIDERS.update({op:fixed_pool for op in ('AvgPool1dFwdOp','AvgPool2dFwdOp','AvgPool3dFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp')})
PROVIDERS.update({op:adaptive_pool for op in ('AdaptiveAvgPool2dFwdOp','AdaptiveMaxPool2dFwdOp')})
PROVIDERS['MeanPoolingFwdOp']=mean_chunks

def dense_dispatch(case):
    return (dense if case.inputs[0].shape[1]==1 else cropped_dense)(case)

def paged_dispatch(case):
    return (paged if case.inputs[0].shape[0]==case.inputs[4].numel() else cropped_paged)(case)

PROVIDERS.update(GQADenseFwdOp=dense_dispatch,GQAPagedFwdOp=paged_dispatch)

from .bit_layout import neg, boolean_binary
PROVIDERS["NegFwdOp"]=neg
PROVIDERS.update({op:boolean_binary for op in ("BitwiseAndFwdOp","BitwiseOrFwdOp","LogicalAndFwdOp","LogicalOrFwdOp")})

from .fma import binary as fma_binary
PROVIDERS.update(MulFwdOp=fma_binary, SubFwdOp=fma_binary)

from .merge_native import arithmetic as merged_arithmetic, selection as merged_selection
PROVIDERS.update({op:merged_arithmetic for op in ('ExpFwdOp','LogFwdOp','SqrtFwdOp','RsqrtFwdOp','BitwiseAndFwdOp','BitwiseOrFwdOp','MulFwdOp','SubFwdOp')})
PROVIDERS.update({op:merged_selection for op in ('WhereFwdOp','MaskedFillScalarFwdOp','MaskedFillTensorFwdOp')})

from .truth import logical as native_logical, extrema as truth_extrema, count_nonzero
_bool_logical={op:PROVIDERS[op] for op in ('LogicalNotFwdOp','LogicalAndFwdOp','LogicalOrFwdOp')}
def logical_dispatch(case):
    name=case._call.signature.name
    return (_bool_logical[name] if all(str(t.dtype)=='torch.bool' for t in case.inputs) else native_logical)(case)
PROVIDERS.update({op:logical_dispatch for op in _bool_logical})
PROVIDERS.update(AllFwdOp=truth_extrema,AnyFwdOp=truth_extrema,CountNonzeroFwdOp=count_nonzero)

from .leaky import leaky
PROVIDERS['LeakyReluFwdOp']=leaky

from .composed import OPERATIONS as COMPOSED_OPERATIONS, composed
PROVIDERS.update({op:composed for op in COMPOSED_OPERATIONS})

from .moe_activation import moe_grouped_dispatch
PROVIDERS['MoEGroupedGemmFwdOp']=moe_grouped_dispatch

from .integration700_capture3 import install as _install_capture3_700
_install_capture3_700(PROVIDERS)

from .integration700_isnan_unpack1 import install as _install_threehost_isnan_unpack1_700
_install_threehost_isnan_unpack1_700(PROVIDERS)

# Two original complete BF16 Bmm contracts, sole server1 agent.
from .single_agent_server1_bmm import install as _install_single_server1_bmm
_install_single_server1_bmm(PROVIDERS)
