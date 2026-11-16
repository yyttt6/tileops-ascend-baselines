"""Vector L1/L2/infinity norms composed solely from original native API outputs."""
from .extended import native, result
from .pointwise import _scale_native
from .reductions import _layout


def vector_norm(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    order = case.params['ord']
    if order not in (1, 2, float("inf")):
        raise UnsupportedCase('This original native route supplies L1/L2/infinity only.')
    axes, kept, width, rows, shape = _layout(case)
    dtype = getattr(torch, case.params['dtype']) if case.params.get('dtype') else case.inputs[0].dtype
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    top = native('moe.topk_gate_kernel', 'topk_gate')
    reducer = native('mhc.norm_fn_kernel', 'mhc_reduce_partials_and_rmsnorm_fwd')
    tokens = (rows + 23) // 24

    def run(x):
        xf = x.float()
        if order in (1, float("inf")):
            negative = _scale_native(torch, product, xf, -1.)
            pair = torch.stack((xf.reshape(-1), negative.reshape(-1)), -1).contiguous()
            elements = pair.gather(1, top(pair, 1)).reshape(x.shape)
        else:
            flat = torch.nn.functional.pad(xf.contiguous().reshape(-1), (0, -x.numel() % 256))
            squares = torch.empty_like(flat).reshape(1, -1)
            product(flat.reshape(1, -1), flat, squares)
            elements = squares.reshape(-1)[:x.numel()].reshape(x.shape)
        if order == float("inf"):
            # All original columns occur exactly once; zero padding is safe for |x|.
            packed = elements.permute(*(kept + axes)).contiguous().reshape(rows, width)
            groups = (width + 1023) // 1024
            grouped = torch.nn.functional.pad(packed, (0, groups * 1024 - width)).reshape(-1, 1024)
            maxima = grouped.gather(1, top(grouped, 1)).reshape(rows, groups).contiguous()
            value = maxima.gather(1, top(maxima, 1))
            return value.to(dtype).reshape(shape)
        values = elements.permute(*(kept + axes)).contiguous().reshape(rows, width).transpose(0, 1)
        partials = torch.nn.functional.pad(values, (0, tokens * 24 - rows)).contiguous().reshape(width, tokens, 1, 24)
        auxiliary = torch.zeros((width, tokens, 1), device=x.device, dtype=torch.float32)
        raw = torch.empty((tokens, 1, 24), device=x.device, dtype=torch.float32)
        sq = torch.empty((tokens, 1), device=x.device, dtype=torch.float32)
        normalized = torch.empty((tokens, 24), device=x.device, dtype=torch.float32)
        reducer(partials, auxiliary, raw, sq, normalized, 1, 1., width)
        value = raw.reshape(-1)[:rows]
        if order == 2:
            # Preserve sqrt(0)=0: native select replaces only denominator zero by1.
            indicator_pair = torch.stack((torch.zeros_like(value), value), -1).contiguous()
            nonzero = top(indicator_pair, 1)
            denominator_pair = torch.stack((torch.ones_like(value), value), -1)
            denominator = denominator_pair.gather(1, nonzero).contiguous().reshape(1, rows, 1)
            numerator = value.reshape(1, rows, 1, 1).expand(1, rows, 1, 24).contiguous()
            raw_root = torch.empty((rows, 1, 24), device=x.device, dtype=torch.float32)
            sq_root = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
            root = torch.empty((rows, 24), device=x.device, dtype=torch.float32)
            reducer(numerator, denominator, raw_root, sq_root, root, 1, 0., 1)
            value = root[:, 0]
        return value.to(dtype).reshape(shape)

    return result(run, 'mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd + moe.topk_gate_kernel.topk_gate + mhc_reduce_partials_and_rmsnorm_fwd',
                  'Exact input widening. Original native absolute-value selection or product square; whole original reduction maps once to native split axis; original raw sum gives L1. Original mHC one-split numerator/square-sum output gives L2 sqrt, with native zero-denominator selection. Infinity uses complete1024-column native TopK groups and native TopK of group maxima. No kernel schedule changes or Torch arithmetic.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py',
                  limitations=('Finite original real inputs; L1/L2/infinity; original comparison tolerance unchanged.',))
