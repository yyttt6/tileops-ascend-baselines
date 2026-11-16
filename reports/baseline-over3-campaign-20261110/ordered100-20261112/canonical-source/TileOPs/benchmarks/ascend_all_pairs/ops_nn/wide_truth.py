"""Exact FP64/complex128 truth through byte layouts and original p=0 APIs."""
from math import prod
import sys
from .common import candidate,new_binding,op_name


def dispatch(case):
    import torch
    if case.inputs[0].dtype in (torch.float64,torch.complex128):return wide_truth(case)
    if op_name(case)=='CountNonzeroFwdOp':
        from .normalization import count_nonzero
        return count_nonzero(case)
    from .reductions import segment
    return segment(case)


def wide_truth(case):
    import ctypes as C
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    if sys.byteorder!='little':raise UnsupportedCase('Inspected byte layout requires little-endian IEEE754 storage.')
    p=case.params;name=op_name(case);binding=new_binding()
    api={'AllFwdOp':'UnsortedSegmentMin','AnyFwdOp':'UnsortedSegmentMax'}.get(name)
    def run(x):
        # The seven low bytes retain every magnitude bit. The high byte is
        # placed in a half's high byte; its sign bit stays a sign, so both
        # signed zeros map to zero, while any exponent/fraction bit is nonzero.
        # Half -> FP32 is exact, including nonzero infinities/NaNs for p=0.
        raw=x.contiguous().view(torch.uint8).reshape(-1,8)
        high=torch.cat((torch.zeros_like(raw[:,:1]),raw[:,7:8]),dim=1).contiguous().view(torch.float16).float()
        lanes=torch.cat((raw[:,:7].float(),high),dim=1).reshape(x.numel(),-1)
        counts=torch.empty((x.numel(),),device=x.device,dtype=torch.float32)
        binding.call('LpNormV2',[lanes],[counts],{'p':0.0,'axes':[1],'keepdim':False,'epsilon':0.0})
        truth=torch.empty_like(counts)
        binding.call('LpNormV2',[counts.reshape(-1,1)],[truth],{'p':0.0,'axes':[1],'keepdim':False,'epsilon':0.0})
        truth=truth.reshape(x.shape)
        dim=p.get('dim');dims=tuple(range(x.ndim)) if dim is None else (dim,) if isinstance(dim,int) else tuple(dim)
        dims=tuple(d%x.ndim for d in dims)
        if api is None:
            if prod(x.shape[d] for d in dims)>2**24:raise UnsupportedCase('FP32 p=0 count exceeds exact integer range.')
            out=torch.empty(tuple(p['output'][0]),device=x.device,dtype=torch.float32)
            binding.call('LpNormV2',[truth],[out],{'p':0.0,'axes':list(dims),'keepdim':False,'epsilon':0.0})
            return out.to(torch.int64)
        outer=tuple(d for d in range(x.ndim) if d not in dims);width=prod(x.shape[d] for d in dims);rows=x.numel()//width
        packed=truth.permute(*dims,*outer).contiguous().reshape(width,rows)
        ids=torch.zeros(width,device=x.device,dtype=torch.int32);number=torch.tensor(1,device=x.device,dtype=torch.int32)
        out=torch.empty((1,rows),device=x.device,dtype=torch.float32)
        binding.call(api,[packed,ids,number],[out],{},constants={2:(C.c_int32*1)(1)})
        return out.reshape(tuple(p['output'][0])).to(torch.bool)
    dirs=('norm/lp_norm_v2',)+(() if api is None else ('index/unsorted_segment_'+('min' if name=='AllFwdOp' else 'max'),))
    return candidate(binding,run,'LpNormV2'+(' + '+api if api else ''),
                     'Lossless IEEE754 byte layout retains all magnitude bits, preserves both signed zeros, and maps high sign bytes to FP16 sign bits. Original p=0 computes truth, then original full-axis p=0/min/max computes the manifest result; all input transforms are charged per call.',*dirs)
