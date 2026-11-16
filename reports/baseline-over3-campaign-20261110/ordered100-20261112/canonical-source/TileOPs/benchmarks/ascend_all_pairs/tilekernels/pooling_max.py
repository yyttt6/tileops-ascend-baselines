"""Complete max-pool windows through original TopK API selection."""
from .extended import native, result


def max_pool(case):
    import itertools
    import math
    import re
    import torch
    name, p = case._call.signature.name, case.params
    rank = int(re.search(r'([123])d', name).group(1))
    unbatched = case.inputs[0].ndim == rank + 1
    spatial = tuple(case.inputs[0].shape[-rank:])
    shape = tuple(case._call.tensors['output'][0])
    output_spatial = shape[-rank:]
    size = math.prod(spatial)
    def seq(v): return (v,) * rank if isinstance(v, int) else tuple(v)
    kernel = seq(p.get('kernel_size', 1))
    stride = seq(p.get('stride') or kernel)
    padding = seq(p.get('padding', 0))
    dilation = seq(p.get('dilation', 1))
    windows = []
    for output_coord in itertools.product(*(range(d) for d in output_spatial)):
        if name.startswith('Adaptive'):
            axes = [range(o*d//z, ((o+1)*d+z-1)//z) for o,d,z in zip(output_coord, spatial, output_spatial)]
        else:
            axes = [[o*s-pad+k*dil for k in range(ks)] for o,s,pad,ks,dil in zip(output_coord,stride,padding,kernel,dilation)]
        window = []
        for coord in itertools.product(*axes):
            if all(0 <= pos < d for pos,d in zip(coord,spatial)):
                index = 0
                for pos,d in zip(coord,spatial): index = index*d+pos
                window.append(index)
        windows.append(window)
    width = max(map(len, windows))
    # Complete original windows; only ragged windows duplicate valid source positions.
    # Earlier asynchronous faults were reproduced in the reference before vendor calls.
    k = 1
    for window in windows: window.extend([window[0]] * (width-len(window)))
    lookup = torch.tensor(windows, device=case.inputs[0].device, dtype=torch.int64)
    top = native('moe.topk_gate_kernel', 'topk_gate')
    def run(x):
        flat = x.float().contiguous().reshape(-1,size)
        values = flat[:,lookup].contiguous().reshape(-1,width)
        # Explicit producer handoff is included in the adapter's execution cost.
        if x.device.type == 'npu': torch.npu.synchronize()
        selected = top(values,k)[:,:1]
        output = values.gather(1,selected).reshape(shape).to(x.dtype)
        if 'Indices' not in name: return output
        ids = lookup.expand(flat.shape[0],-1,-1).reshape(-1,width).gather(1,selected).reshape(shape)
        return output, ids
    return result(run, 'moe.topk_gate_kernel.topk_gate(k=1)',
                  'Gather each complete pooling window in source order; ragged windows pad with their first valid source index. Synchronize the packed-input producer before the original native call; this compatibility barrier is timed. No reduction split or kernel changes. All native work and value packing are timed; gather original values/indices and restore manifest shape including unbatched input.',
                  'moe/topk_gate_kernel.py','moe/topk_gate_asc.py',
                  limitations=('Finite manifest values. Padding duplicates an existing source index; original stable topk chooses first maximal source index.',))
