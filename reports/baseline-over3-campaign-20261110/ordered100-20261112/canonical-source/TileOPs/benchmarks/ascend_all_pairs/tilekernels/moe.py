"""Sigmoid routing assembled from original TileKernels routing primitives."""
from .extended import native, result
from .pointwise import _scale_native


def fused_topk(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    p = case.params
    if p['scoring_func'] != 'sigmoid':
        raise UnsupportedCase('Original routing is sqrtsoftplus; native sigmoid mix output can supply sigmoid scores, but no equivalent standalone softmax output exists.')
    sigmoid_fn = native('mhc.pre_split_mixes_kernel', 'mhc_pre_split_mixes_fwd')
    top = native('moe.topk_gate_kernel', 'topk_gate')
    normalize = native('moe.normalize_weight_kernel', 'normalize_weight')
    add = native('quant.norm_forward_kernel', 'norm_forward')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    x = case.inputs[0]
    groups = p['num_expert_group']
    group_width = x.shape[-1] // groups
    scale = torch.ones(3, device=x.device, dtype=torch.float32)
    bias0 = torch.zeros(24, device=x.device, dtype=torch.float32)
    expert_grid = torch.arange(x.shape[-1], device=x.device, dtype=torch.int64).reshape(1, groups, group_width)

    def sum_native(left, right):
        shape = left.shape
        count = left.numel()
        a = torch.nn.functional.pad(left.contiguous().reshape(-1), (0, -count % 256)).reshape(-1, 256)
        b = torch.nn.functional.pad(right.expand(shape).contiguous().reshape(-1), (0, -count % 256)).reshape(-1, 256)
        return add(a, None, 1e-6, residual=b)[2].reshape(-1)[:count].reshape(shape)

    def run(logits, correction_bias=None):
        count = logits.numel()
        values = torch.nn.functional.pad(logits.float().contiguous().reshape(-1), (0, -count % 256)).reshape(-1, 4)
        packed = torch.nn.functional.pad(values, (0, 20))
        pre = torch.empty_like(values)
        post = torch.empty_like(values)
        comb = torch.empty((values.shape[0], 16), device=logits.device, dtype=torch.float32)
        sigmoid_fn(packed, scale, bias0, pre, post, comb, 1., 0.)
        scores = pre.reshape(-1)[:count].reshape_as(logits)
        selection = scores if correction_bias is None else sum_native(scores, correction_bias)
        if groups > 1:
            grouped = selection.reshape(-1, group_width).contiguous()
            group_top = top(grouped, 1 if correction_bias is None else 2)
            group_values = grouped.gather(1, group_top)
            group_scores = group_values[:, 0] if correction_bias is None else sum_native(group_values[:, 0], group_values[:, 1])
            chosen_groups = top(group_scores.reshape(logits.shape[0], groups).contiguous(), p['topk_group'])
            group_index = chosen_groups[:, :, None].expand(-1, -1, group_width)
            chosen = selection.reshape(-1, groups, group_width).gather(1, group_index).reshape(logits.shape[0], -1)
            original_ids = expert_grid.expand(logits.shape[0], -1, -1).gather(1, group_index).reshape(logits.shape[0], -1)
            ids = original_ids.gather(1, top(chosen.contiguous(), p['top_k']))
        else:
            ids = top(selection.contiguous(), p['top_k'])
        weights = scores.gather(1, ids).contiguous()
        if p['renormalize']:
            weights = normalize(weights)[1]
        if p['routed_scaling_factor'] != 1.:
            weights = _scale_native(torch, product, weights, p['routed_scaling_factor'])
        return weights, ids.to(torch.int32)

    return result(run, 'mhc.pre_split_mixes_kernel.mhc_pre_split_mixes_fwd + moe.topk_gate_kernel.topk_gate + moe.normalize_weight_kernel.normalize_weight + optional quant.norm_forward_kernel.norm_forward[2] / mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd',
                  'Select original sigmoid scores; original residual-sum output supplies optional expert bias and top-two group sum. Original top-k selects groups/experts, gather maps original expert indices; native normalize_weight and native product supply renormalization/scaling. No Torch arithmetic; all input-dependent packing inside run.',
                  'mhc/pre_split_mixes_kernel.py', 'mhc/pre_split_mixes_asc.py',
                  'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py',
                  'moe/normalize_weight_kernel.py', 'moe/normalize_weight_asc.py',
                  'quant/norm_forward_kernel.py', 'quant/norm_forward_asc.py',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py',
                  limitations=('Sigmoid manifest routing only; original normalize_weight adds1e-20 to its strictly-positive denominator, below FP32 resolution for these manifest scores.',))
