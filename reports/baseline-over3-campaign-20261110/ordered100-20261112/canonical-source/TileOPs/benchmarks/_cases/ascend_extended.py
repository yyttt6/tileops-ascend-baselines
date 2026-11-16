"""Additional manifest factories, shared by normal and Ascend benchmarks."""

from benchmarks._cases import Entry
from benchmarks.baselines import private_inputs
from workloads.ascend_extended import (
    CausalConvCall, CausalConvUpdateCall, Int8GemmCall, MergeAttentionCall, PagedCacheGatherCall,
    SelectiveScanCall, FusedQKNormRoPECall, FusedAddRMSNormQuantCall,
)
from workloads.ascend_extended_attention import MLAKVCacheWriteCall, MLAPagedCall, MLAVarlenCall
from workloads.ascend_extended_dsa import DSAPagedCall
from workloads.ascend_extended_transform_quant import GemmW4A8Call, HadamardTransformCall


def _causal_binder(op, case):
    def result(*args):
        output = op(*args)
        return output, args[2]

    return private_inputs(result, case.inputs, 2)


def _gather_binder(op, case):
    def result(*args):
        op(*args)
        return args[0]

    return private_inputs(result, case.inputs, 0)


def _qk_rope_binder(op, case):
    def result(*args):
        op(*args)
        return args[0]

    return private_inputs(result, case.inputs, 0)


def _mla_cache_write_binder(op, case):
    def result(*args):
        op(*args)
        return args[2]

    return private_inputs(result, case.inputs, 2)


ENTRIES = {
    "CausalConv1dFwdOp": Entry(CausalConvCall, binder=_causal_binder),
    "CausalConv1dUpdateFwdOp": Entry(CausalConvUpdateCall, binder=_causal_binder),
    "PagedKVCacheGatherFwdOp": Entry(PagedCacheGatherCall, binder=_gather_binder),
    "GemmINT8W8A8FwdOp": Entry(Int8GemmCall),
    "MergeAttentionStatesFwdOp": Entry(MergeAttentionCall),
    "SelectiveScanFwdOp": Entry(SelectiveScanCall),
    "FusedQKNormRoPEFwdOp": Entry(FusedQKNormRoPECall, binder=_qk_rope_binder),
    "FusedAddRMSNormPerTokenQuantFwdOp": Entry(FusedAddRMSNormQuantCall),
    "MLAPagedFwdOp": Entry(MLAPagedCall),
    "MLAVarlenFwdOp": Entry(MLAVarlenCall),
    "MLAKVCacheWriteFwdOp": Entry(MLAKVCacheWriteCall, binder=_mla_cache_write_binder),
    "DSAPagedFwdOp": Entry(DSAPagedCall),
    "HadamardTransformFwdOp": Entry(HadamardTransformCall),
    "GemmW4A8FwdOp": Entry(GemmW4A8Call),
}
