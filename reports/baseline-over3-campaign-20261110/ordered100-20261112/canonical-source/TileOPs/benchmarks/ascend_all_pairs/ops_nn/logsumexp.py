"""Recover logsumexp from original LogSoftmax at its maximal input position."""
from math import prod
from .common import candidate,new_binding


def factory(case):
    import ctypes as C
    from dataclasses import replace
    from pathlib import Path
    import sys
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
    from . import ieee_lookup
    if sys.byteorder!='little':
        raise UnsupportedCase('Exact IEEE masks require the inspected little-endian layout.')
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise UnsupportedCase('Original FP32 LogSoftmax mapping covers FP16/BF16/FP32 inputs.')
    p=case.params;binding=new_binding()

    def run(x):
        dims=p.get('dim');dims=tuple(range(x.ndim)) if dims is None else (dims,) if isinstance(dims,int) else tuple(dims)
        dims=tuple(d%x.ndim for d in dims);outer=tuple(d for d in range(x.ndim) if d not in dims)
        width=prod(x.shape[d] for d in dims)
        if not 0<width<2**31:
            raise UnsupportedCase('The original max-pool index must fit INT32 and the reduced axis must be nonempty.')
        xx=x.permute(*outer,*dims).contiguous().reshape(-1,width).float();rows=xx.shape[0]
        maxima=torch.empty((rows,1,1,1,1),device=x.device,dtype=torch.float32)
        index=torch.empty(maxima.shape,device=x.device,dtype=torch.int32)
        binding.call('AdaptiveMaxPool3d',[xx.reshape(rows,1,1,1,width)],[maxima,index],
                     {'output_size':[1,1,1],'indices_dtype':3})
        normalized=torch.empty_like(xx)
        binding.call('LogSoftmaxV2',[xx],[normalized],{'axes':[1]})
        selected=torch.empty((rows,1),device=x.device,dtype=torch.float32)
        binding.call('GatherElements',[normalized,index.reshape(rows,1)],[selected],{'dim':1})
        result=torch.empty((rows,),device=x.device,dtype=torch.float32)
        alpha=torch.tensor(1.0,device=x.device,dtype=torch.float32)
        binding.call('ForeachSubList',[maxima.reshape(rows),selected.reshape(rows),alpha],[result],{},
                     names=['x10','x20','alpha','y0'],constants={2:(C.c_float*1)(1.0)})

        # LogSoftmax's indeterminate infinity subtraction must not turn a
        # mathematically infinite logsumexp into NaN. Input NaNs take precedence.
        # Every predicate is computed by original own Gather/arithmetic APIs.
        axis=torch.tensor(0,device=x.device,dtype=torch.int32)
        tables={}
        def halves(value):
            raw=value.contiguous().reshape(-1).view(torch.uint8).reshape(-1,4)
            zero=torch.zeros_like(raw[:,0])
            return (torch.stack((raw[:,2],raw[:,3],zero,zero),1).contiguous().view(torch.int32).reshape(-1),
                    torch.stack((raw[:,0],raw[:,1],zero,zero),1).contiguous().view(torch.int32).reshape(-1))
        def lookup(kind,indices):
            if kind not in tables:
                tables[kind]=torch.tensor(ieee_lookup.table_values('bfloat16',kind),device=x.device,dtype=torch.uint8)
            out=torch.empty(indices.shape,device=x.device,dtype=torch.uint8)
            binding.call('GatherV2',[tables[kind],indices,axis],[out],
                         {'batch_dims':0,'negative_index_support':False},constants={2:(C.c_int32*1)(0)})
            return out.float()
        def binary(api,a,b):
            out=torch.empty_like(a)
            binding.call(api,[a,b],[out],{},names=['x10','x20','y0'])
            return out
        high,low=halves(xx)
        input_nan=binary('ForeachMaximumList',lookup('isnan',high),
                         binary('ForeachMulList',lookup('isinf',high),lookup('low_nonzero',low)))
        nan_count=torch.empty((rows,),device=x.device,dtype=torch.float32)
        binding.call('LpNormV2',[input_nan.reshape(rows,width)],[nan_count],
                     {'p':0.0,'axes':[1],'keepdim':False,'epsilon':0.0})
        nan_flag=torch.empty_like(nan_count)
        binding.call('LpNormV2',[nan_count.reshape(rows,1)],[nan_flag],
                     {'p':0.0,'axes':[1],'keepdim':False,'epsilon':0.0})
        high,low=halves(maxima);low_zero=lookup('low_zero',low)
        masks=(binary('ForeachMulList',lookup('positive_inf',high),low_zero),
               binary('ForeachMulList',lookup('negative_inf',high),low_zero),nan_flag)
        for mask,value in zip(masks,(float('inf'),float('-inf'),float('nan'))):
            updates=torch.full((rows,),value,device=x.device,dtype=torch.float32)
            out=torch.empty_like(result)
            binding.call('MaskedScatter',[result,mask.to(torch.bool),updates],[out],{})
            result=out
        return result.to(getattr(torch,p['output'][1])).reshape(tuple(p['output'][0]))

    result=candidate(binding,run,
                     'AdaptiveMaxPool3d + LogSoftmaxV2 + GatherElements + ForeachSubList + GatherV2 + ForeachMulList + ForeachMaximumList + LpNormV2 + MaskedScatter',
                     'Pack complete reduction axes per row and widen exactly to FP32. Select original LogSoftmax at the original maximum index and recover max-logsoftmax(max) by native subtraction, avoiding cancellation at arbitrary input positions. Original lookup/arithmetic/p=0/masked-scatter handles input NaNs and both infinite maxima. All layouts, tables, masks and output casts are inside run; no Torch reduction/logarithm/predicate arithmetic.',
                     'pooling/adaptive_max_pool3d','activation/log_softmax_v2','index/gather_elements',
                     'foreach/foreach_sub_list','index/gather_v2','foreach/foreach_mul_list',
                     'foreach/foreach_maximum_list','norm/lp_norm_v2','index/masked_scatter')
    return replace(result,source_sha256=result.source_sha256|source_hashes(Path(ieee_lookup.__file__)))
