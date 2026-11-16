"""Original exported normalization outputs and grouped expert-selection API."""
from pathlib import Path


def _tracked(candidate):
    from dataclasses import replace
    from .hashing import source_hashes
    return replace(candidate, source_sha256=candidate.source_sha256 | source_hashes(Path(__file__)))


def softmax_dispatch(case):
    if case.inputs[0].shape[case.params['dim']] <= 1024:
        from .adapters import softmax
        return _tracked(softmax(case))
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    from .adapters import _candidate
    from .native import GraphCall
    x, p = case.inputs[0], case.params
    dim = p['dim'] % x.ndim
    dtype = getattr(torch, p['dtype']) if p.get('dtype') else x.dtype
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or dtype not in (x.dtype, torch.float32):
        raise UnsupportedCase('Original full-softmax output only permits unchanged input dtype or lossless FP32 widening.')
    binding = GraphCall('ops-transformer', vendor_name='selectionpairs_transformer')
    def run(x):
        moved = x.movedim(dim, -1)
        width = moved.shape[-1]
        rows = moved.float().contiguous().reshape(-1, width)
        outputs = []
        for begin in range(0, len(rows), 128):
            inp = rows[begin:begin + 128]
            selected = torch.empty((len(inp), 1), dtype=torch.float32, device=x.device)
            ids = torch.empty_like(selected, dtype=torch.int32)
            full = torch.empty_like(inp)
            binding.call('MoeGatingTopKSoftmaxV2', [inp, None], [selected, ids, full],
                         {'k': 1, 'renorm': 0, 'output_softmax_result_flag': True})
            outputs.append(full)
            binding.cleanup()
        return torch.cat(outputs).reshape(moved.shape).movedim(-1, dim).to(dtype).contiguous()
    return _tracked(_candidate(binding, run, 'MoeGatingTopKSoftmaxV2', 'moe/moe_gating_top_k_softmax_v2',
        'Move complete reduction axis, losslessly widen input to FP32, select original full softmax_result output with renorm=0 and k=1. No reduction-axis splitting, probability reconstruction, or external softmax; restore manifest dtype/layout inside run.'))


def sigmoid_selection(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    from .adapters import _candidate
    from .native import GraphCall
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase('Original sigmoid normOut supports floating inputs without narrowing.')
    binding = GraphCall('ops-transformer', vendor_name='selectionpairs_transformer')
    def run(x):
        flat = x.contiguous().flatten().float()
        padded = torch.nn.functional.pad(flat, (0, (-flat.numel()) % 1024)).reshape(-1, 1024)
        outputs = []
        for begin in range(0, len(padded), 1024):
            inp = padded[begin:begin + 1024]
            selected = torch.empty((len(inp), 1), dtype=torch.float32, device=x.device)
            ids = torch.empty_like(selected, dtype=torch.int32)
            full = torch.empty_like(inp)
            binding.call('MoeGatingTopK', [inp, None, None, None], [selected, ids, full],
                         {'k': 1, 'k_group': 1, 'group_count': 1, 'group_select_mode': 0,
                          'renorm': 0, 'norm_type': 1, 'out_flag': True,
                          'routed_scaling_factor': 1., 'eps': 0.})
            outputs.append(full)
            binding.cleanup()
        return torch.cat(outputs).flatten()[:x.numel()].reshape_as(x).to(x.dtype)
    return _tracked(_candidate(binding, run, 'MoeGatingTopK', 'moe/moe_gating_top_k',
        'Original norm_type=1 out_flag=True exports complete FP32 sigmoid before expert selection/weight renormalization. Lossless input widening, independent pointwise packing/padding/crop and output cast occur inside run.'))


def fused_topk_dispatch(case):
    p = case.params
    if ((p['scoring_func'] == 'softmax' and p['num_expert_group'] == 1 and
         p['topk_group'] == 1 and p['routed_scaling_factor'] == 1.) or
        (p['scoring_func'] == 'sigmoid' and not p['renormalize'])):
        from .routing import routing
        return _tracked(routing(case))
    import torch
    from .adapters import _candidate
    from .native import GraphCall
    binding = GraphCall('ops-transformer', vendor_name='selectionpairs_transformer')
    def run(gating_output, correction_bias=None):
        inp = gating_output.float().contiguous()
        bias = None if correction_bias is None else correction_bias.float().contiguous()
        weights = torch.empty((len(inp), p['top_k']), dtype=torch.float32, device=inp.device)
        ids = torch.empty_like(weights, dtype=torch.int32)
        full = torch.empty_like(inp)
        binding.call('MoeGatingTopK', [inp, bias, None, None], [weights, ids, full],
                     {'k': p['top_k'], 'k_group': p['topk_group'], 'group_count': p['num_expert_group'],
                      'group_select_mode': 1 if bias is not None else 0, 'renorm': int(p['renormalize']),
                      'norm_type': 0 if p['scoring_func'] == 'softmax' else 1, 'out_flag': False,
                      'routed_scaling_factor': float(p['routed_scaling_factor']), 'eps': 0.})
        return weights, ids
    return _tracked(_candidate(binding, run, 'MoeGatingTopK', 'moe/moe_gating_top_k',
        'Original grouped top-k API: lossless FP32 logits/bias, group max or top-two-sum selection, native scoring/normalization/scaling with eps=0 matching manifest. Complete weights/IDs oracle and original tie acceptance retained.'))
