"""Original native sum reducer for the FP32 contract sensitive to GEMM order."""
from math import prod
from pathlib import Path
from .microblocks import _load_builder


def sum_fwd(case):
    # A single original full-row call remains valid; recursive reductions do not.
    _shape = case.params['x'][0]
    _dims = case.params.get('dim')
    _dims = tuple(range(len(_shape))) if _dims is None else ((_dims,) if isinstance(_dims, int) else tuple(_dims))
    _width = prod(_shape[d % len(_shape)] for d in _dims)
    if str(case.inputs[0].dtype) == 'torch.float32' and _width > 128:
        raise RuntimeError("Original-only boundary: repeated partial-sum composition is retired; original full-row API support remains unresolved.")
    import torch
    if case.inputs[0].dtype!=torch.float32:
        from .adapters import sum_mean
        return sum_mean(case)
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    x,=case.inputs
    dims=case.params.get('dim')
    dims=tuple(range(x.ndim)) if dims is None else tuple(d%x.ndim for d in ((dims,) if isinstance(dims,int) else dims))
    outer=tuple(d for d in range(x.ndim) if d not in dims)
    order=outer+dims
    m,k=prod(x.shape[d] for d in outer),prod(x.shape[d] for d in dims)
    shape=tuple(1 if d in dims else x.shape[d] for d in range(x.ndim)) if case.params.get('keepdim',False) else tuple(x.shape[d] for d in outer)
    output_dtype=getattr(torch,case.params['dtype']) if case.params.get('dtype') else x.dtype
    tl,builder,hashes=_load_builder('testing/ascend/language/test_tilelang_ascend_reduce.py','make_default_layout_reduce_kernel')
    hashes.update(source_hashes(Path(__file__)))
    kernel=tl.compile(builder(2,128,'float32','sum'),target='ascend',pass_configs={tl.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE:False})
    def run(x):
        data=x.permute(order).contiguous().reshape(m,k)
        while True:
            width=data.shape[-1]
            packed=F.pad(data,(0,-width%128)).reshape(-1,128)
            count=packed.shape[0]
            packed=F.pad(packed,(0,0,0,count%2))
            result=torch.empty(packed.shape[0],device=x.device,dtype=data.dtype)
            for start in range(0,packed.shape[0],2):kernel(packed[start:start+2],result[start:start+2])
            data=result[:count].reshape(m,-1)
            if data.shape[-1]==1:break
        return data.reshape(shape).to(output_dtype)
    return Candidate(Implementation(run),'tilelang::make_default_layout_reduce_kernel(2,128,sum)',
                     'Original native sum reducer and original pass configuration; axis packing, neutral padding and repeated original reducer calls combine partial sums. No Torch reduction, no tolerance or kernel-body edits.',hashes)
