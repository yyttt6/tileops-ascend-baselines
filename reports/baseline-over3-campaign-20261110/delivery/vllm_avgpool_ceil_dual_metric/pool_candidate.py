from pathlib import Path
from math import prod
from itertools import product
from .runtime import TRITON_ROOT,candidate,module,validate
def pool_original(case):
    validate(case)
    import torch
    import torch.nn.functional as F
    mean, sources = module(TRITON_ROOT + 'batch_invariant/mean.py')
    p, op = (case.params, case._call.signature.name)
    divisor = p.get('divisor_override')
    if divisor is not None:
        scale, extra = module(TRITON_ROOT + 'muls_add.py', hardware=True)
        sources += extra
    adaptive = op == 'AdaptiveAvgPool2dFwdOp'
    ndim = len(p['input'][0]) - 2

    def tup(value):
        return (value,) * ndim if isinstance(value, int) else tuple(value)

    def run(x):
        if adaptive and x.ndim == 3:
            x = x.unsqueeze(0)
        shape = x.shape[2:]
        output_spatial = p['output'][0][-2:] if adaptive else p['output'][0][2:]
        groups = {}
        if not adaptive:
            kernel, padding = (tup(p['kernel_size']), tup(p['padding']))
            stride = tup(p['stride'] or p['kernel_size'])
            include = p['count_include_pad']
            data = F.pad(x, tuple((v for pad in reversed(padding) for v in (pad, pad)))) if include else x
        else:
            data = x
        for flat, pos in enumerate(product(*(range(n) for n in output_spatial))):
            if adaptive:
                bounds = [(i * n // o, ((i + 1) * n + o - 1) // o) for i, n, o in zip(pos, shape, output_spatial)]
            elif include:
                bounds = [(i * s, min(i * s + k, n + 2 * pad)) for i, s, k, n, pad in zip(pos, stride, kernel, shape, padding)]
            else:
                bounds = [(max(0, i * s - pad), min(n, i * s - pad + k)) for i, s, k, n, pad in zip(pos, stride, kernel, shape, padding)]
            width = prod((end - begin for begin, end in bounds))
            groups.setdefault(width, []).append((flat, tuple((slice(begin, end) for begin, end in bounds))))
        result = torch.empty((x.shape[0], x.shape[1], prod(output_spatial)), device=x.device, dtype=x.dtype)
        for width, windows in groups.items():
            block = max(1, 1048576 // (x.shape[0] * x.shape[1] * width))
            for begin in range(0, len(windows), block):
                part = windows[begin:begin + block]
                packed = torch.stack([data[slice(None), slice(None), *s].contiguous().reshape(x.shape[0], x.shape[1], width) for _, s in part], dim=2)
                avg = mean.mean_dim(packed.reshape(-1, width).float(), 1, dtype=torch.float32)
                if divisor is not None:
                    avg = scale.muls_add_triton(avg.reshape(-1, 1), torch.zeros_like(avg).reshape(-1, 1), width / divisor)
                ids = torch.tensor([i for i, _ in part], device=x.device, dtype=torch.long)
                result[:, :, ids] = avg.reshape(x.shape[0], x.shape[1], len(part)).to(x.dtype)
        return result.reshape(p['output'][0])
    return candidate(run, 'mean_dim (spatial windows)', 'Slice complete regular/adaptive windows, retaining explicit padding and exact ceil-tail divisor rules; batch independent equal-cardinality windows into original FP32 mean. Optional divisor override uses original scalar multiplication. All padding, packing, launches and scatter are timed.', [Path(__file__), *sources], timing_mode='host_completed')

def pool_fixed(case):
    validate(case)
    import torch
    mean,sources=module(TRITON_ROOT+'batch_invariant/mean.py')
    p=case.params; ndim=len(p['input'][0])-2
    def tup(v):return (v,)*ndim if isinstance(v,int) else tuple(v)
    kernel=tup(p['kernel_size']);stride=tup(p['stride'] or p['kernel_size']);padding=tup(p['padding'])
    assert case._call.signature.name in ('AvgPool1dFwdOp','AvgPool3dFwdOp')
    assert not p['count_include_pad'] and p.get('divisor_override') is None
    def run(x):
        shape=x.shape[2:];outsp=p['output'][0][2:];groups={}
        for flat,pos in enumerate(product(*(range(n) for n in outsp))):
            bounds=[(max(0,i*s-pad),min(n,i*s-pad+k)) for i,s,k,n,pad in zip(pos,stride,kernel,shape,padding)]
            indices=[sum(v*prod(shape[a+1:]) for a,v in enumerate(coords)) for coords in product(*(range(begin,end) for begin,end in bounds))]
            assert indices
            groups.setdefault(len(indices),[]).append((flat,indices))
        result=torch.empty((x.shape[0],x.shape[1],prod(outsp)),device=x.device,dtype=x.dtype)
        data=x.reshape(x.shape[0],x.shape[1],prod(shape))
        for count,windows in groups.items():
            block=max(1,1048576//(x.shape[0]*x.shape[1]*count))
            for start in range(0,len(windows),block):
                part=windows[start:start+block]
                indices=torch.tensor([v for _,window in part for v in window],device=x.device,dtype=torch.long)
                packed=data.index_select(2,indices).reshape(-1,count).float()
                avg=mean.mean_dim(packed,1,dtype=torch.float32)
                ids=torch.tensor([i for i,_ in part],device=x.device,dtype=torch.long)
                result[:,:,ids]=avg.reshape(x.shape[0],x.shape[1],len(part)).to(x.dtype)
        return result.reshape(p['output'][0])
    return candidate(run,'mean_dim (clipped indexed windows)','Original clipped bounds and true window denominator, same groups/block. Per-call index metadata and all pack/FP32/scatter timed.',[Path(__file__),*sources],timing_mode='host_completed')
