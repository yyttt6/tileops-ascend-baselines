"""Device-resident, ordinary PyTorch implementations for eleven missing baselines.

These candidates are independent of the CPU workload oracles. Importing this
package does not import TileLang, torch_npu, or any external baseline provider.
"""

from functools import partial

from benchmarks.pytorch_missing.attention import (
    dsa_paged,
    mla_cache_write,
    mla_paged,
    paged_cache_gather,
)
from benchmarks.pytorch_missing.linear import deltanet_chunk_bwd, gla_chunk_bwd, selective_scan
from benchmarks.pytorch_missing.pointwise import fused_add_rms_quant, fused_qk_norm_rope, hadamard
from benchmarks.pytorch_missing.w4a8 import gemm_w4a8, repack_w4a8

__all__ = ["FUNCTIONS", "bind", "implementation", "repack_w4a8"]

FUNCTIONS = {
    "DeltaNetChunkBwdOp": deltanet_chunk_bwd,
    "GLAChunkBwdOp": gla_chunk_bwd,
    "GemmW4A8FwdOp": gemm_w4a8,
    "DSAPagedFwdOp": dsa_paged,
    "FusedAddRMSNormPerTokenQuantFwdOp": fused_add_rms_quant,
    "FusedQKNormRoPEFwdOp": fused_qk_norm_rope,
    "HadamardTransformFwdOp": hadamard,
    "MLAKVCacheWriteFwdOp": mla_cache_write,
    "MLAPagedFwdOp": mla_paged,
    "PagedKVCacheGatherFwdOp": paged_cache_gather,
    "SelectiveScanFwdOp": selective_scan,
}


def bind(op, params):
    """Bind manifest scalar parameters; all tensors remain call-time inputs.

    Cache gather, QK norm/RoPE and cache write implement the declared mutation
    and return the mutated tensor for verification. Benchmark callers must use
    the usual private-input/reset policy for those three entries.
    """
    return partial(FUNCTIONS[op], **params)


def implementation(case):
    """An opt-in benchmark Implementation with the standard mutation/reset policy.

    This leaves the shared provider registry untouched. W4A8 callers must opt in
    to the format documented in w4a8.py and use the matching independent oracle;
    the legacy Case.reference is still explicitly unavailable.
    """
    from benchmarks.api import Implementation
    from benchmarks.baselines import private_inputs

    op = case._call.signature.name
    run = bind(op, case._call.params)
    mutated = {"FusedQKNormRoPEFwdOp": 0, "MLAKVCacheWriteFwdOp": 2, "PagedKVCacheGatherFwdOp": 0}
    if op in mutated:
        return private_inputs(run, case.inputs, mutated[op])
    return Implementation(run=run)
