"""Original matrix APIs for fixed-coefficient linear maps and reductions."""
from pathlib import Path

def linear(case):
    import torch
    from dataclasses import replace
    from .adapters import _candidate,_float_product,_gemm
    from .native import GraphCall as FullGraph
    from benchmarks.ascend_graph_call import GraphCall as HalfGraph
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
    from .hashing import source_hashes
    x=case.inputs[0];op=case._call.signature.name
    if x.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise UnsupportedCase('Original native matrix map has no lossless integer input widening.')
    full=x.dtype==torch.float32;binding=(FullGraph if full else HalfGraph)('ops-transformer')
    alpha=case.params.get('alpha',1)
    if alpha!=1:raise UnsupportedCase('Only exact unit subtraction coefficient is currently mapped.')
    weight=case.params.get('weight',None) if op=='LerpScalarFwdOp' else None
    if weight is not None and any(float(torch.tensor(c,dtype=x.dtype).item())!=c for c in (1-weight,weight)):
        raise UnsupportedCase('Original input dtype cannot represent the scalar interpolation coefficients exactly.')
    def run(x,y=None):
        if y is None:packed=x.contiguous().reshape(-1,1);coeff=[-1.];shape=x.shape
        else:
            a,b=torch.broadcast_tensors(x,y);shape=a.shape
            packed=torch.stack((a,b),-1).reshape(-1,2).contiguous();coeff=[1-weight,weight] if weight is not None else [1.,-1.]
        matrix_weight=torch.tensor(coeff,dtype=x.dtype,device=x.device)[:,None]
        out=_float_product(binding,packed,matrix_weight) if full else _gemm(binding,packed,matrix_weight,x.dtype)
        return out.reshape(shape).to(x.dtype)
    result=_candidate(binding,run,'GroupedMatmul' if full else 'GroupedMatmulAdd','gmm/grouped_matmul' if full else 'gmm/grouped_matmul_add','Broadcast/pack one or two scalar columns; original native GEMM with exactly representable -1, [1,-1], or [1-weight,weight] constants performs negation, subtraction or scalar interpolation. No input-dependent arithmetic outside original matrix API; all packing and output casts timed.')
    return replace(result,source_sha256=result.source_sha256|source_hashes(Path(__file__)))

def reduction_dispatch(case):
    import torch,math
    from dataclasses import replace
    from .adapters import reduction,_candidate,_float_product
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
    from .hashing import source_hashes
    x=case.inputs[0]
    if x.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise UnsupportedCase('Original FP32 matrix reduction requires lossless floating input widening.')
    p=case.params;ds=p['dim'];ds=range(x.ndim) if ds is None else ((ds,) if isinstance(ds,int) else ds)
    dims=sorted(d%x.ndim for d in ds);kept=[i for i in range(x.ndim) if i not in dims]
    width=math.prod(x.shape[d] for d in dims);mean=case._call.signature.name=='MeanFwdOp'
    if x.dtype!=torch.float32 and (not mean or width&(width-1)==0):return reduction(case)
    binding=GraphCall('ops-transformer')
    def run(x):
        packed=x.permute(kept+dims).contiguous().float().reshape(-1,width)
        weight=torch.full((width,1),1/width if mean else 1.,dtype=torch.float32,device=x.device)
        out=_float_product(binding,packed,weight)
        dtype=getattr(torch,p['dtype']) if p.get('dtype') else x.dtype
        shape=[1 if i in dims else size for i,size in enumerate(x.shape)] if p['keepdim'] else [x.shape[i] for i in kept]
        return out.to(dtype).reshape(shape)
    result=_candidate(binding,run,'GroupedMatmul','gmm/grouped_matmul','Preserve FP32 inputs or losslessly widen half inputs for non-power-of-two means; use the original native FP32 matrix API for full reduction; FP32 unit or inverse-width coefficients, reduction-axis packing and keepdim restoration are timed.')
    return replace(result,source_sha256=result.source_sha256|source_hashes(Path(__file__)))
