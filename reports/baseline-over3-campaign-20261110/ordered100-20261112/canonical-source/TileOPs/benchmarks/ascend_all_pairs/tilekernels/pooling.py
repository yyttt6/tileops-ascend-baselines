"""Whole pooling windows mapped to original native sum/top-one reductions."""
from .extended import native, result


def pooling(case):
    import torch
    import itertools
    import math
    name, p = case._call.signature.name, case.params
    is_sequence = name == 'MeanPoolingFwdOp'
    maximum = 'MaxPool' in name
    indices_out = 'Indices' in name
    top = native('moe.topk_gate_kernel', 'topk_gate') if maximum else None
    reduce = native('mhc.norm_fn_kernel', 'mhc_reduce_partials_and_rmsnorm_fwd') if not maximum else None
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd') if not maximum else None
    x = case.inputs[0]
    if not is_sequence:
        spatial = tuple(x.shape[2:])
        rank = len(spatial)
        output_shape = tuple(case._call.tensors['output'][0])
        output_spatial = output_shape[2:]
        size = math.prod(spatial)
        def seq(v): return (v,) * rank if isinstance(v, int) else tuple(v)
        kernel = seq(p.get('kernel_size', 1))
        stride = seq(p.get('stride') or kernel)
        padding = seq(p.get('padding', 0))
        dilation = seq(p.get('dilation', 1))
        windows, counts = [], []
        for out_coord in itertools.product(*(range(d) for d in output_spatial)):
            if name.startswith('Adaptive'):
                axes = [range(o * d // z, ((o + 1) * d + z - 1) // z)
                        for o, d, z in zip(out_coord, spatial, output_spatial)]
            else:
                axes = [[o * s - pad + k * dil for k in range(ks)]
                        for o, s, pad, ks, dil in zip(out_coord, stride, padding, kernel, dilation)]
            window = []
            divisor = 0
            for coord in itertools.product(*axes):
                valid = all(0 <= pos < d for pos, d in zip(coord, spatial))
                if valid:
                    flat = 0
                    for pos, d in zip(coord, spatial): flat = flat * d + pos
                    window.append(flat)
                else: window.append(size)
                in_padding = all(-pad <= pos < d + pad for pos, d, pad in zip(coord, spatial, padding))
                divisor += valid or (bool(p.get('count_include_pad')) and in_padding)
            if maximum:
                first_valid = next(index for index in window if index != size)
                window = [first_valid if index == size else index for index in window]
            windows.append(window)
            counts.append(p.get('divisor_override') or divisor)
        width = max(map(len, windows))
        if maximum: width = (width + 63) // 64 * 64
        for window in windows: window.extend([window[0] if maximum else size] * (width - len(window)))
        lookup = torch.tensor(windows, device=x.device, dtype=torch.int64)
        reciprocal = torch.tensor([1. / n for n in counts], device=x.device, dtype=torch.float32)

    def average(values, scale):
        rows, width = values.shape
        tokens = (rows + 23) // 24
        partials = torch.nn.functional.pad(values.transpose(0, 1), (0, tokens * 24 - rows)).contiguous().reshape(width, tokens, 1, 24)
        aux = torch.zeros((width, tokens, 1), device=values.device, dtype=torch.float32)
        raw = torch.empty((tokens, 1, 24), device=values.device, dtype=torch.float32)
        sq = torch.empty((tokens, 1), device=values.device, dtype=torch.float32)
        normalized = torch.empty((tokens, 24), device=values.device, dtype=torch.float32)
        reduce(partials, aux, raw, sq, normalized, 1, 1., width)
        left = torch.nn.functional.pad(raw.reshape(-1)[:rows], (0, -rows % 256))
        right = torch.nn.functional.pad(scale.contiguous().reshape(-1), (0, -rows % 256))
        out = torch.empty_like(left).reshape(1, -1)
        product(left.reshape(1, -1), right, out)
        return out.reshape(-1)[:rows]

    def run(x, offsets=None, indices=None):
        if is_sequence:
            chunk = p['chunk_size']
            if offsets is None:
                slices = [(start, min(start + chunk, x.shape[1])) for start in range(0, x.shape[1], chunk)]
            else:
                # Metadata determines only layout and divisors; copied every call and included in timing.
                bounds = offsets.tolist()
                slices = [(bounds[s] + i * chunk, min(bounds[s] + (i + 1) * chunk, bounds[s + 1])) for s, i in indices.tolist()]
            windows = [list(range(a, b)) + [x.shape[1]] * (chunk - b + a) for a, b in slices]
            lookup_run = torch.tensor(windows, device=x.device, dtype=torch.int64)
            padded = torch.cat((x.float(), torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=torch.float32)), 1)
            values = padded[:, lookup_run].permute(0, 1, 3, 4, 2).contiguous()
            out_shape = values.shape[:-1]
            scales = torch.tensor([1. / (b-a) for a,b in slices], device=x.device, dtype=torch.float32).reshape(1, -1, 1, 1).expand(out_shape)
            return average(values.reshape(-1, chunk), scales).reshape(out_shape).to(x.dtype)
        flat = x.float().contiguous().reshape(-1, size)
        padded = flat if maximum else torch.nn.functional.pad(flat, (0, 1), value=0.)
        values = padded[:, lookup].contiguous().reshape(-1, width)
        if maximum:
            selected = top(values, 1)
            output = values.gather(1, selected).reshape(output_shape).to(x.dtype)
            if not indices_out: return output
            ids = lookup.expand(flat.shape[0], -1, -1).reshape(-1, width).gather(1, selected).reshape(output_shape)
            return output, ids
        scales = reciprocal.expand(flat.shape[0], -1)
        return average(values, scales).reshape(output_shape).to(x.dtype)
    api = 'moe.topk_gate_kernel.topk_gate(k=1)' if maximum else 'mhc.norm_fn_kernel.mhc_reduce_partials_and_rmsnorm_fwd[out_mul] + mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd'
    return result(run, api,
                  'Enumerate original full pooling windows from shape/stride/dilation/padding metadata; max windows pad to64-column alignment by replicating each window first valid source index; gather is layout only. Original native top-one supplies maxima and source indices, or original full split sum and native reciprocal-count product supply averages. Ragged sequence metadata is read inside run; all value packing/cropping is timed.',
                  'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py', 'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py',
                  limitations=('Finite manifest inputs; max padding duplicates a valid value from its own window, preserving maxima and original source indices without synthetic sentinels.',))
