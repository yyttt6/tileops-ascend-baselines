"""Original native complete split sums for wide logical and integer-count cases."""
from .extended import native, result
from .pointwise import _scale_native
from .reductions import _layout


def complete_logical(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    axes, kept, width, rows, shape = _layout(case)
    dtype = case.inputs[0].dtype
    if dtype not in (torch.bool, torch.int32, torch.int64) or width > 2**24:
        raise UnsupportedCase('Complete logical sum requires bool or exact int16 lane views and count <=2**24.')
    top = native('moe.topk_gate_kernel', 'topk_gate')
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd') if dtype != torch.bool else None
    reducer = native('mhc.norm_fn_kernel', 'mhc_reduce_partials_and_rmsnorm_fwd')
    tokens = (rows + 23) // 24
    name = case._call.signature.name

    def run(x):
        if x.dtype == torch.bool:
            bits = x.float()
        else:
            # Every signed integer is zero iff all its 16-bit words are zero.
            # The bit view is lossless; all int16 words widen exactly to FP32.
            lanes = x.contiguous().view(torch.int16).reshape(x.numel(), x.element_size() // 2).float()
            negative = _scale_native(torch, product, lanes, -1.)
            scores = torch.cat((lanes, negative), dim=1).contiguous()
            magnitude = scores.gather(1, top(scores, 1))
            pair = torch.cat((torch.zeros_like(magnitude), magnitude), dim=1).contiguous()
            bits = top(pair, 1).float().reshape(x.shape)
        values = bits.permute(*(kept + axes)).contiguous().reshape(rows, width).transpose(0, 1)
        partials = torch.nn.functional.pad(values, (0, tokens * 24 - rows)).contiguous().reshape(width, tokens, 1, 24)
        auxiliary = torch.zeros((width, tokens, 1), device=x.device, dtype=torch.float32)
        raw = torch.empty((tokens, 1, 24), device=x.device, dtype=torch.float32)
        sq = torch.empty((tokens, 1), device=x.device, dtype=torch.float32)
        normalized = torch.empty((tokens, 24), device=x.device, dtype=torch.float32)
        # The original native API reduces the ENTIRE requested axis, once.
        reducer(partials, auxiliary, raw, sq, normalized, 1, 1., width)
        count = raw.reshape(-1)[:rows]
        if name == 'CountNonzeroFwdOp':
            return count.to(torch.int64).reshape(shape)
        if name == 'AnyFwdOp':
            pair = torch.stack((torch.zeros_like(count), count), dim=-1).contiguous()
            return top(pair, 1).to(torch.bool).reshape(shape)
        # Stable first-index tie rule: count==width selects label true.
        pair = torch.stack((count, torch.full_like(count, width)), dim=-1).contiguous()
        labels = torch.tensor([1, 0], device=x.device, dtype=torch.uint8)
        return labels[top(pair, 1)].to(torch.bool).reshape(shape)

    return result(run, 'mhc.norm_fn_kernel.mhc_reduce_partials_and_rmsnorm_fwd[out_mul] + moe.topk_gate_kernel.topk_gate + optional mhc_fn_normw_merge_fwd',
                  'Lossless int16 word views and exact widening; native maximum over signed words gives per-integer nonzero. Original native split-sum covers the entire original reduction, exact 0/1 count <=2**24. Native stable top-one converts complete counts to All/Any. No Torch arithmetic or split reduction.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py',
                  limitations=('Only bool/int32/int64 complete counts <=2**24; unused native normalization outputs retained.',))


def all_any(case):
    import torch
    from .additional import numerical_boolean
    x = case.inputs[0]
    if x.dtype in (torch.int32, torch.int64) or (x.dtype == torch.bool and _layout(case)[2] >= 11008):
        return complete_logical(case)
    return numerical_boolean(case)


def count_nonzero(case):
    import torch
    from .select_extra import extra
    if case.inputs[0].dtype in (torch.int32, torch.int64):
        return complete_logical(case)
    return extra(case)
