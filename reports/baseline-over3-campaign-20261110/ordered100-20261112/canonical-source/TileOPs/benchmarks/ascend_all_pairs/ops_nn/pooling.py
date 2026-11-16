"""Pooling with lossless rank/layout adaptations and original graph attributes."""
from math import prod
from .common import candidate, new_binding, op_name
from .catalog import POOL


def _tuple(value, n):
    return (value,)*n if isinstance(value,int) else tuple(value)


def pool(case):
    import torch
    name=op_name(case);api,directory=POOL[name];p=case.params;binding=new_binding()
    output_key='output' if 'output' in p else 'values'
    out_shape=tuple(p[output_key][0])
    indices_requested='Indices' in name
    def run(x):
        if 'Adaptive' in name:
            size=p['output_size'];size=(size,size) if isinstance(size,int) or size is None else tuple(size)
            size=[x.shape[-2+i] if v is None else int(v) for i,v in enumerate(size)]
            out=torch.empty(out_shape,device=x.device,dtype=x.dtype)
            if 'Max' in name:
                xx=x.contiguous().unsqueeze(0) if x.ndim==3 else x.contiguous()
                native_out=out.unsqueeze(0) if x.ndim==3 else out
                indices=torch.empty_like(native_out,dtype=torch.int64)
                binding.call(api,[xx],[native_out,indices],{'output_size':size,'argmax_dtype':9})
                indices=indices.reshape(out_shape)
                return (out,indices) if indices_requested else out
            widened=torch.empty(out_shape,device=x.device,dtype=torch.float32)
            binding.call(api,[x.float().contiguous()],[widened],{'output_size':size})
            return widened.to(x.dtype)
        n=int(name.split('Pool')[1][0]);missing=3-n
        # Native 3D kernels cover 1D/2D by singleton spatial axes; flattened indices stay identical.
        batched=x.ndim==n+2
        base_shape=tuple(x.shape) if batched else (1,*x.shape)
        base_out=out_shape if batched else (1,*out_shape)
        in_shape=(*base_shape[:2],*(1,)*missing,*base_shape[2:])
        native_out=(*base_out[:2],*(1,)*missing,*base_out[2:])
        xx=x.contiguous().reshape(in_shape)
        k=(1,)*missing+_tuple(p['kernel_size'],n)
        stride=(1,)*missing+_tuple(p['stride'] if p['stride'] is not None else p['kernel_size'],n)
        pad=(0,)*missing+_tuple(p['padding'],n)
        out=torch.empty(native_out,device=x.device,dtype=x.dtype)
        if 'Max' in name:
            idx=torch.empty_like(out,dtype=torch.int64)
            attrs={'ksize':list(k),'strides':list(stride),'pads':list(pad),
                   'dilation':list((1,)*missing+_tuple(p['dilation'],n)),
                   'ceil_mode':bool(p['ceil_mode']),'data_format':'NCDHW','dtype':9}
            binding.call(api,[xx],[out,idx],attrs,formats=[30,30,30])
            return (out.reshape(out_shape),idx.reshape(out_shape)) if indices_requested else out.reshape(out_shape)
        attrs={'ksize':[1,1,*k],'strides':[1,1,*stride],
               'pads':[v for v in pad for _ in (0,1)],'ceil_mode':bool(p['ceil_mode']),
               'count_include_pad':bool(p['count_include_pad']),
               'divisor_override':int(p.get('divisor_override') or 0),'data_format':'NCDHW'}
        binding.call(api,[xx],[out],attrs,formats=[30,30])
        return out.reshape(out_shape)
    return candidate(binding,run,api,'Exact pooling attributes, per-call contiguous packing; 1D/2D use singleton axes of original 3D kernel; all requested indices retained.',directory)


def reduction_pool(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    name=op_name(case);p=case.params;binding=new_binding()
    api='AdaptiveAvgPool2d' if name=='MeanFwdOp' else 'AdaptiveMaxPool3d'
    directory='pooling/adaptive_avg_pool2d' if name=='MeanFwdOp' else 'pooling/adaptive_max_pool3d'
    out_shape=tuple(p['output'][0])
    def run(x):
        dims=p.get('dim');dims=tuple(range(x.ndim)) if dims is None else (dims,) if isinstance(dims,int) else tuple(dims)
        dims=tuple(d%x.ndim for d in dims);outer=tuple(d for d in range(x.ndim) if d not in dims)
        rows=prod(x.shape[d] for d in outer);width=prod(x.shape[d] for d in dims)
        dtype=getattr(torch,p['dtype']) if p.get('dtype') else x.dtype
        xx=x.permute(*outer,*dims).contiguous().reshape(rows,1,1,width).to(dtype)
        if name=='MeanFwdOp':
            xx=xx.float();out=torch.empty((rows,1,1,1),device=x.device,dtype=torch.float32)
            binding.call(api,[xx],[out],{'output_size':[1,1]})
            return out.reshape(out_shape).to(dtype)
        xx=xx.reshape(rows,1,1,1,width)
        out=torch.empty((rows,1,1,1,1),device=x.device,dtype=x.dtype)
        idx=torch.empty_like(out,dtype=torch.int32)
        binding.call(api,[xx],[out,idx],{'output_size':[1,1,1],'indices_dtype':3})
        return (idx.to(torch.int64) if name=='ArgmaxFwdOp' else out).reshape(out_shape)
    return candidate(binding,run,api,'Pack complete reduction axes into one spatial window; preserve outer rows, keepdim and output dtype; select original value or flattened index output.',directory)


def chunk_mean(case):
    import torch
    from dataclasses import replace
    from pathlib import Path
    from benchmarks.ascend_baselines import source_hashes
    binding=new_binding()
    def run(x,offsets=None,indices=None):
        slices=case.workload.chunk_slices(x,offsets,indices)
        outputs=[]
        for start,end in slices:
            xx=x[:,start:end].permute(0,2,3,1).contiguous().reshape(-1,1,1,end-start).float()
            out=torch.empty((xx.shape[0],1,1,1),device=x.device,dtype=torch.float32)
            binding.call('AdaptiveAvgPool2d',[xx],[out],{'output_size':[1,1]})
            outputs.append(out.reshape(x.shape[0],x.shape[2],x.shape[3]).to(x.dtype))
        return torch.stack(outputs,dim=1) if outputs else x.new_empty((x.shape[0],0,*x.shape[2:]))
    result=candidate(binding,run,'AdaptiveAvgPool2d','Each complete logical chunk becomes one original adaptive-average window; read ragged offsets/indices afresh each invocation, preserve caller order and partial-chunk divisor.','pooling/adaptive_avg_pool2d')
    return replace(result,timing_mode='host_roundtrip' if len(case.inputs)>1 else 'device_async',
                   source_sha256=result.source_sha256|source_hashes(Path(case.workload.chunk_slices.__code__.co_filename)))
