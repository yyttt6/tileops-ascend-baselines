"""CPU fixtures/oracles and unchanged numerical policies, separate from candidates."""

import torch

from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from workloads.ascend_extended import (
    FusedAddRMSNormQuantCall,
    FusedQKNormRoPECall,
    PagedCacheGatherCall,
    SelectiveScanCall,
)
from workloads.ascend_extended_attention import MLAKVCacheWriteCall, MLAPagedCall
from workloads.ascend_extended_dsa import DSAPagedCall
from workloads.ascend_extended_transform_quant import GemmW4A8Call, HadamardTransformCall
from workloads.linear_attention.chunk_cpu import deltanet_saved_cpu, gla_chunk_states_cpu
from workloads.linear_attention.deltanet import DeltaNetChunkwiseCall
from workloads.linear_attention.gla import GLAChunkwiseCall
from workloads.numerics import Exact

WORKLOADS = {
    "DeltaNetChunkBwdOp": DeltaNetChunkwiseCall,
    "GLAChunkBwdOp": GLAChunkwiseCall,
    "GemmW4A8FwdOp": GemmW4A8Call,
    "DSAPagedFwdOp": DSAPagedCall,
    "FusedAddRMSNormPerTokenQuantFwdOp": FusedAddRMSNormQuantCall,
    "FusedQKNormRoPEFwdOp": FusedQKNormRoPECall,
    "HadamardTransformFwdOp": HadamardTransformCall,
    "MLAKVCacheWriteFwdOp": MLAKVCacheWriteCall,
    "MLAPagedFwdOp": MLAPagedCall,
    "PagedKVCacheGatherFwdOp": PagedCacheGatherCall,
    "SelectiveScanFwdOp": SelectiveScanCall,
}

MUTATED = {"FusedQKNormRoPEFwdOp": 0, "MLAKVCacheWriteFwdOp": 2, "PagedKVCacheGatherFwdOp": 0}


def calls(op):
    plan = entry_plan(op, load_manifest()[op], load_adts(), resolve=False)
    return [
        instantiate(plan, row, dt)
        for row in load_workloads(op)
        for dt in row.get("dtype_cases") or [{}]
    ]


def workload(op, call):
    result = WORKLOADS[op](call, device="cpu")
    # These eleven signatures contain no constructor tensors. Avoid an extra
    # multi-GB random materialization merely to read scalar arguments.
    result.arguments = lambda: call.arguments({})
    return result


def inputs_for(op, w):
    inputs = w.gen_inputs()
    if op == "DeltaNetChunkBwdOp":
        do, q, k, v, beta, *_ = inputs
        inputs = (
            do,
            q,
            k,
            v,
            beta,
            *deltanet_saved_cpu(q, k, v, beta, w.call.params["chunk_size"]),
        )
    elif op == "GLAChunkBwdOp":
        q, k, v, g, h, do, dht = inputs
        initial = h[:, 0] if w.call.params["has_initial_state"] else None
        h = gla_chunk_states_cpu(k, v, g, w.call.params["chunk_size"], initial)
        inputs = q, k, v, g, h, do, torch.zeros_like(dht)
    return inputs


def w4a8_reference(a, packed, group, channel, token, *, group_size):
    """Independent CPU nibble decoder + FP64 contraction, with FP8 weight rounding.

    Arithmetic follows the upstream semantic reference. BF16 group input is the
    explicitly documented compatibility extension for the local twelve cases.
    This oracle is never called by a device candidate.
    """
    m, k = a.shape
    n = packed.shape[0]
    result = torch.empty((m, n), dtype=torch.bfloat16)
    activation = a.double()
    lut = torch.tensor(list(range(8)) + list(range(-8, 0)), dtype=torch.float64)
    for begin in range(0, n, 256):
        raw = packed[begin : begin + 256].long()
        weight = torch.empty((raw.shape[0], k), dtype=torch.float64)
        weight[:, 0::2] = lut[raw.remainder(16)]
        weight[:, 1::2] = lut[raw.div(16, rounding_mode="floor")]
        weight *= group[begin : begin + raw.shape[0]].double().repeat_interleave(group_size, 1)
        weight = weight.to(torch.float8_e4m3fn).double()
        dot = (activation @ weight.T).float()
        result[:, begin : begin + raw.shape[0]] = (
            dot * token[:, None] * channel[None, begin : begin + raw.shape[0]]
        ).to(torch.bfloat16)
    return result


def reference(op, w, inputs):
    if op == "GemmW4A8FwdOp":
        return w4a8_reference(*inputs, group_size=w.call.params["group_size"])
    return w.ref_program(*inputs)


def evidence(op, w, inputs):
    if op == "GemmW4A8FwdOp":
        return Exact(atol=5e-5 * inputs[0].shape[1], rtol=2e-2, normalized=5e-5)
    return w.verification(*inputs)
