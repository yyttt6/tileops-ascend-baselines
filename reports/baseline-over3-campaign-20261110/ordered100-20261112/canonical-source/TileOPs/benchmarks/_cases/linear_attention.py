"""Case factories of the linear_attention family."""

import torch

from benchmarks._cases import Entry
from workloads.linear_attention.chunk_cpu import deltanet_saved_cpu, gla_chunk_states_cpu
from workloads.linear_attention.deltanet import (
    DeltaNetChunkwiseCall,
    DeltaNetDecodeCall,
    DeltaNetFwdCall,
)
from workloads.linear_attention.gdn import GDNFwdCall
from workloads.linear_attention.gla import GLAChunkwiseCall, GLADecodeCall, GLAFwdCall
from workloads.linear_attention.kda import KDAFwdCall


def _deltanet_bwd_inputs(workload: DeltaNetChunkwiseCall) -> tuple:
    """Forward-consistent saved inputs from independent CPU mathematics."""
    do, q, k, v, beta, *_saved = workload.gen_inputs()
    saved = deltanet_saved_cpu(
        *(t.detach().cpu() for t in (q, k, v, beta)), workload.arguments()["chunk_size"]
    )
    return do, q, k, v, beta, *(t.to(q.device) for t in saved)


def _gla_bwd_inputs(workload: GLAChunkwiseCall) -> tuple:
    """Independent CPU chunk states; retain the existing zero-dht case policy."""
    q, k, v, g, _h, do, _dht = workload.gen_inputs()
    arguments = workload.arguments()
    initial = _h[:, 0].detach().cpu() if arguments["has_initial_state"] else None
    h = gla_chunk_states_cpu(
        *(t.detach().cpu() for t in (k, v, g)), arguments["chunk_size"], initial
    ).to(q.device)
    dht = torch.zeros_like(_dht)
    return q, k, v, g, h, do, dht


ENTRIES = {
    "DeltaNetFwdOp": Entry(DeltaNetFwdCall),
    "DeltaNetChunkFwdOp": Entry(DeltaNetChunkwiseCall),
    "DeltaNetChunkBwdOp": Entry(DeltaNetChunkwiseCall, inputs=_deltanet_bwd_inputs),
    "DeltaNetRecurrentFwdOp": Entry(DeltaNetDecodeCall),
    "GDNFwdOp": Entry(GDNFwdCall),
    "GLAChunkFwdOp": Entry(GLAChunkwiseCall),
    "GLAChunkBwdOp": Entry(GLAChunkwiseCall, inputs=_gla_bwd_inputs),
    "GLAFwdOp": Entry(GLAFwdCall),
    "GLARecurrentFwdOp": Entry(GLADecodeCall),
    "KDAFwdOp": Entry(KDAFwdCall),
}
