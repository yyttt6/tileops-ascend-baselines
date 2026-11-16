"""Reduction-to-segment/index API mappings, without altered native math."""
from math import prod
from .common import candidate,new_binding,op_name


def segment(case):
    import ctypes as C
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    name=op_name(case);p=case.params;binding=new_binding()
    api={'SumFwdOp':'UnsortedSegmentSum','ProdFwdOp':'UnsortedSegmentProd','AminFwdOp':'UnsortedSegmentMin',
         'AllFwdOp':'UnsortedSegmentMin','AnyFwdOp':'UnsortedSegmentMax'}[name]
    logical=name in ('AllFwdOp','AnyFwdOp')
    from .truth import truth01
    numeric=logical and case.inputs[0].dtype!=torch.bool
    if logical and case.inputs[0].dtype==torch.complex128:raise UnsupportedCase('No exact FP64 complex truth specialization in original p=0 API.')
    def run(x):
        dims=p.get('dim');dims=tuple(range(x.ndim)) if dims is None else (dims,) if isinstance(dims,int) else tuple(dims)
        dims=tuple(d%x.ndim for d in dims);outer=tuple(d for d in range(x.ndim) if d not in dims)
        width=prod(x.shape[d] for d in dims);rows=x.numel()//width
        dtype=torch.float32 if logical else getattr(torch,p['output'][1])
        if logical:x=truth01(binding,x)
        xx=x.permute(*dims,*outer).contiguous().reshape(width,rows).to(dtype)
        ids=torch.zeros(width,device=x.device,dtype=torch.int32);num=torch.tensor(1,device=x.device,dtype=torch.int32)
        out=torch.empty((1,rows),device=x.device,dtype=dtype)
        binding.call(api,[xx,ids,num],[out],{},constants={2:(C.c_int32*1)(1)})
        return out.reshape(tuple(p['output'][0])).to(torch.bool) if logical else out.reshape(tuple(p['output'][0]))
    snake={'UnsortedSegmentSum':'sum','UnsortedSegmentProd':'prod','UnsortedSegmentMin':'min','UnsortedSegmentMax':'max'}[api]
    return candidate(binding,run,api+(' + LpNormV2' if numeric else ''),'Pack all reduction axes as the segment axis, use a single original segment per output coordinate; preserve explicit output dtype and keepdim. Boolean min/max consumes exact native p=0 truth values.','index/unsorted_segment_'+snake,'norm/lp_norm_v2')


def argmin(case):
    import torch
    p=case.params;binding=new_binding()
    def run(x):
        dim=p.get('dim');xx=x.contiguous().reshape(1,-1) if dim is None else x.movedim(dim,-1).contiguous()
        shape=(*xx.shape[:-1],1);values=torch.empty(shape,device=x.device,dtype=x.dtype);indices=torch.empty(shape,device=x.device,dtype=torch.int64)
        binding.call('KthValue',[xx],[values,indices],{'k':1,'dim':-1})
        return indices.reshape(tuple(p['output'][0]))
    return candidate(binding,run,'KthValue(k=1)','Pack complete argmin axis last, select original first order statistic index, restore declared keepdim shape; exact tie behavior remains subject to manifest validation.','index/kth_value')


def rsqrt(case):
    import ctypes as C
    import torch
    binding=new_binding()
    def run(x):
        dtype=torch.float16 if x.dtype==torch.float16 else torch.float32
        scalar=torch.tensor(-0.5,device=x.device,dtype=dtype)
        constant=(C.c_uint16*1)(0xb800) if dtype==torch.float16 else (C.c_float*1)(-0.5)
        out=torch.empty_like(x)
        binding.call('ForeachPowScalar',[x.contiguous(),scalar],[out],{},names=['x0','scalar','y0'],constants={1:constant})
        return out
    return candidate(binding,run,'ForeachPowScalar(exponent=-0.5)','Select original floating exponent API for reciprocal square root; exact exponent constant and per-call stride packing.','foreach/foreach_pow_scalar')
