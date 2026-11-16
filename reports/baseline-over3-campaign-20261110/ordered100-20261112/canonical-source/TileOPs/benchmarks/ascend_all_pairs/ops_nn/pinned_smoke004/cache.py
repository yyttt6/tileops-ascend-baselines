"""Original index APIs for complete cache gather and nonquantized cache writes."""
from .common import candidate,new_binding,op_name


def _gather(binding,rows,indices):
    import ctypes as C
    import torch
    axis=torch.tensor(0,device=rows.device,dtype=torch.int32)
    out=torch.empty((indices.numel(),*rows.shape[1:]),device=rows.device,dtype=rows.dtype)
    binding.call('GatherV2',[rows.contiguous(),indices.contiguous(),axis],[out],
                 {'batch_dims':0,'negative_index_support':False},constants={2:(C.c_int32*1)(0)})
    return out


def cache_gather(case):
    import torch
    from dataclasses import replace
    binding=new_binding();scaled=case.params.get('scale') is not None
    def run(dst,cache,block_table,offsets,starts=None,scale=None):
        table=block_table.tolist();cu=offsets.tolist();begin=[0]*(len(cu)-1) if starts is None else starts.tolist()
        physical=[];page_size=cache.shape[1]
        for batch,(lo,hi) in enumerate(zip(cu[:-1],cu[1:])):
            physical.extend(int(table[batch][position//page_size])*page_size+position%page_size
                            for position in range(begin[batch],begin[batch]+hi-lo))
        if len(physical)!=dst.shape[0]:raise ValueError('Manifest destination and complete ragged row count disagree')
        ids=torch.tensor(physical,device=cache.device,dtype=torch.int64)
        rows=cache.contiguous().reshape(cache.shape[0]*page_size,-1)
        data=_gather(binding,rows.view(torch.uint8),ids).view(cache.dtype).reshape(dst.shape)
        if scale is not None:
            value=torch.empty(data.shape,device=cache.device,dtype=torch.float32)
            binding.call('ForeachMulList',[data.float(),scale.float().expand(data.shape).contiguous()],[value],{},names=['x10','x20','y0'])
            data=value
        return data.to(dst.dtype)
    result=candidate(binding,run,'GatherV2'+(' + ForeachMulList' if scaled else ''),
                     'Read page/sequence metadata afresh, flatten physical cache rows, original GatherV2 copies every byte including FP8 payloads, optional original native scale multiply, restore complete destination dtype/shape.',
                     'index/gather_v2','foreach/foreach_mul_list')
    return replace(result,timing_mode='host_roundtrip')


def cache_write(case):
    import torch
    from dataclasses import replace
    from benchmarks.ascend_baselines import UnsupportedCase
    name=op_name(case);p=case.params
    if name=='MLAKVCacheWriteFwdOp':
        if p['fuse_rope']:raise UnsupportedCase('Native scatter route does not implement fused RoPE; an original equivalent rotation API is required.')
        if case.inputs[2].dtype!=case.inputs[0].dtype:raise UnsupportedCase('Native scatter route covers exact copy, not FP8 cache quantization.')
    elif case.inputs[2].dtype!=case.inputs[0].dtype:
        raise UnsupportedCase('Native scatter route covers exact copy, not FP8 cache quantization.')
    binding=new_binding()
    def update(cache,values,slots):
        metadata=slots.tolist();tokens=[i for i,s in enumerate(metadata) if s>=0];destinations=[s for s in metadata if s>=0]
        if len(destinations)!=len(set(destinations)):raise UnsupportedCase('Original scatter order is unspecified for duplicate destinations; manifest requires independent rows.')
        ids=torch.tensor(tokens,device=values.device,dtype=torch.int64)
        indices=torch.tensor(destinations,device=values.device,dtype=torch.int64)
        result=cache.contiguous().clone();flat=result.reshape(cache.shape[0]*cache.shape[1],-1).view(torch.uint8)
        from math import prod
        updates=values.index_select(0,ids).contiguous().reshape(len(tokens),prod(values.shape[1:])).view(torch.uint8)
        binding.call('ScatterUpdate',[flat,indices,updates],[flat],{'use_locking':False})
        return result
    def run(*inputs):
        if name=='MLAKVCacheWriteFwdOp':
            latent,pe,cache,slots,*unused=inputs
            return update(cache,torch.cat((latent,pe),dim=-1),slots)
        k,v,k_cache,v_cache,slots,*unused=inputs
        return update(k_cache,k,slots),update(v_cache,v,slots)
    result=candidate(binding,run,'ScatterUpdate',
                     'Clone complete cache pools, read slot metadata afresh, select valid update rows as per-call input layout, original ScatterUpdate writes full row bytes; every untouched byte and original input is preserved. FP8 quantization and fused RoPE excluded.',
                     'index/scatter_update')
    return replace(result,timing_mode='host_roundtrip')
