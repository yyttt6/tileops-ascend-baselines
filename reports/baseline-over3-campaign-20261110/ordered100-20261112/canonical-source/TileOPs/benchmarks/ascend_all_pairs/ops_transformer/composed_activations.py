"""Native weighted sums and raw-score selection compose exact activation formulas."""
from pathlib import Path


def composed_activation(case):
    import torch
    from dataclasses import replace
    from .native import GraphCall
    from .adapters import _candidate
    from .routing_arithmetic import weighted_rows
    from .hashing import source_hashes
    from benchmarks.ascend_baselines import UnsupportedCase, WORKSPACE
    if any(torch.is_tensor(x) and x.dtype not in (torch.float16, torch.bfloat16, torch.float32) for x in case.inputs):
        raise UnsupportedCase('Only exactly widened floating inputs are bound.')
    op, p = case._call.signature.name, case.params
    routing = GraphCall('ops-transformer')
    selection = GraphCall('ops-transformer', vendor_name='selectionpairs_transformer')

    def weighted(a, b, wa, wb):
        a, b, wa, wb = torch.broadcast_tensors(a, b, wa, wb)
        shape = a.shape
        values = torch.stack((a, b), -1).reshape(-1, 2, 1)
        weights = torch.stack((wa, wb), -1).reshape(-1, 2)
        result = weighted_rows(routing, values, weights).reshape(shape)
        routing.cleanup()
        return result

    def extreme(a, b, minimum=False):
        a, b = torch.broadcast_tensors(a, b)
        packed = torch.stack((a, b), -1).reshape(-1, 2)
        outputs = []
        for begin in range(0, len(packed), 65536):
            source = packed[begin:begin + 65536]
            k = 2 if minimum else 1
            values = torch.empty((len(source), k), dtype=torch.float32, device=a.device)
            ids = torch.empty_like(values, dtype=torch.int32)
            unused = torch.empty((0,), dtype=torch.float32, device=a.device)
            selection.call('MoeGatingTopKSoftmaxV2', [source, None], [values, ids, unused],
                           {'k': k, 'renorm': 1, 'output_softmax_result_flag': False})
            outputs.append(source.gather(1, ids[:, -1:].long()))
            selection.cleanup()
        return torch.cat(outputs).reshape(a.shape)

    def run(x, end_or_weight=None, weight=None):
        z = x.float()
        scalar = lambda v: torch.full((), v, dtype=torch.float32, device=x.device)
        zero, one = scalar(0.), scalar(1.)
        if op == 'LerpTensorFwdOp':
            w = weight.float()
            inverse = weighted(one, w, one, scalar(-1.))
            return weighted(z, end_or_weight.float(), inverse, w).to(x.dtype)
        if op in ('PreluFwdOp', 'LeakyReluFwdOp'):
            low, high = extreme(z, zero, True), extreme(z, zero)
            if op == 'PreluFwdOp':
                slope = end_or_weight.float()
                slope = slope.reshape(()) if slope.numel() == 1 else slope.reshape(1, -1, *([1] * (x.ndim - 2)))
            else:
                slope = scalar(p['negative_slope'])
            return weighted(high, low, one, slope).to(x.dtype)
        shifted = weighted(z, scalar(3.), one, one)
        clipped = extreme(extreme(shifted, zero), scalar(6.), True)
        if op == 'HardswishFwdOp':
            product = weighted_rows(routing, clipped.reshape(-1, 1, 1), z.reshape(-1, 1))
            routing.cleanup()
            clipped = product.reshape(z.shape)
        result = weighted_rows(routing, clipped.reshape(-1, 1, 1), scalar(1 / 6).expand(clipped.numel(), 1).contiguous())
        return result.reshape(x.shape).to(x.dtype)

    api = 'MoeFinalizeRoutingV2' + ('' if op == 'LerpTensorFwdOp' else '+MoeGatingTopKSoftmaxV2')
    candidate = _candidate(routing, run, api, 'moe/moe_finalize_routing_v2',
        'Losslessly widen floating inputs. Original native weighted sums/products and raw-score extrema implement the manifest hard activation, PReLU/LeakyReLU or tensor interpolation formula. All runtime broadcasts/constants/packing/gathers/casts are timed; no Torch activation/product/sum or input-derived cache. Unchanged complete oracle determines numerical acceptance.')
    hashes = source_hashes(Path(__file__), Path(__file__).with_name('routing_arithmetic.py'))
    if op != 'LerpTensorFwdOp':
        hashes |= source_hashes(WORKSPACE / 'ascend_baselines/ops-transformer/moe/moe_gating_top_k_softmax_v2', selection.vendor)
    def cleanup():
        routing.cleanup()
        selection.cleanup()
    return replace(candidate, cleanup=cleanup, source_sha256=candidate.source_sha256 | hashes)
