"""Full FP32 fused SiLU gate through original normalization and weighted products."""
from pathlib import Path


def fused_silu(case):
    import torch
    from dataclasses import replace
    from .adapters import _candidate
    from .native import GraphCall
    from .routing_arithmetic import weighted_rows
    from .hashing import source_hashes
    from benchmarks.ascend_baselines import UnsupportedCase, WORKSPACE
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase('Original fused gate mapping requires lossless floating input widening.')
    norm = GraphCall('ops-transformer', vendor_name='selectionpairs_transformer')
    routing = GraphCall('ops-transformer')
    def run(x):
        half = x.shape[-1] // 2
        gate = x[..., :half].float().contiguous().flatten()
        up = x[..., half:].float().contiguous().flatten()
        padded = torch.nn.functional.pad(gate, (0, (-gate.numel()) % 1024)).reshape(-1, 1024)
        blocks = []
        for start in range(0, len(padded), 1024):
            source = padded[start:start + 1024]
            score = torch.empty((len(source), 1), dtype=torch.float32, device=x.device)
            ids = torch.empty_like(score, dtype=torch.int32)
            full = torch.empty_like(source)
            norm.call('MoeGatingTopK', [source, None, None, None], [score, ids, full],
                      {'k': 1, 'k_group': 1, 'group_count': 1, 'group_select_mode': 0,
                       'renorm': 0, 'norm_type': 1, 'out_flag': True,
                       'routed_scaling_factor': 1., 'eps': 0.})
            blocks.append(full)
            norm.cleanup()
        sigmoid = torch.cat(blocks).flatten()[:gate.numel()]
        activated = weighted_rows(routing, gate.reshape(-1, 1, 1), sigmoid.reshape(-1, 1))
        routing.cleanup()
        result = weighted_rows(routing, activated.reshape(-1, 1, 1), up.reshape(-1, 1))
        return result.reshape(*x.shape[:-1], half).to(x.dtype)
    candidate = _candidate(norm, run, 'MoeGatingTopK+MoeFinalizeRoutingV2', 'moe/moe_gating_top_k',
        'Lossless FP32 gate/up split. Original full sigmoid normOut followed by two original FP32 routing products computes SiLU(gate)*up before the sole manifest output cast. No rounded half activation intermediate, Torch math fallback, or input-dependent caching; complete original fused oracle retained.')
    def cleanup():
        norm.cleanup(); routing.cleanup()
    return replace(candidate, cleanup=cleanup, source_sha256=candidate.source_sha256 | source_hashes(
        Path(__file__), Path(__file__).with_name('routing_arithmetic.py'),
        WORKSPACE / 'ascend_baselines/ops-transformer/moe/moe_finalize_routing_v2', routing.vendor))
