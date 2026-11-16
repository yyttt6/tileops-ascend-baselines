"""IEEE754 classification through original GatherV2 and immutable bit tables."""
from functools import lru_cache
import sys
from .common import candidate,new_binding,op_name


@lru_cache(None)
def table_values(dtype,kind):
    """Input-independent host constants, never input tensor data."""
    result=[]
    for bits in range(65536):
        if dtype=='float16':exponent=(bits>>10)&31;mantissa=bits&1023;full=31
        else:exponent=(bits>>7)&255;mantissa=bits&127;full=255
        if kind=='isfinite':value=exponent!=full
        elif kind=='isinf':value=exponent==full and mantissa==0
        elif kind=='isnan':value=exponent==full and mantissa!=0
        elif kind=='positive_inf':value=bits==0x7f80
        elif kind=='negative_inf':value=bits==0xff80
        elif kind=='low_zero':value=bits==0
        elif kind=='low_nonzero':value=bits!=0
        elif kind=='replace':
            maximum=0x7bff if dtype=='float16' else 0x7f7f
            value=bits if exponent!=full else 0 if mantissa else maximum|(bits&0x8000)
            if value>=32768:value-=65536
        else:raise ValueError(kind)
        result.append(int(value))
    return tuple(result)


def factory(case):
    import ctypes as C
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    if sys.byteorder!='little':raise UnsupportedCase('IEEE lookup requires inspected little-endian storage.')
    dtype=case.inputs[0].dtype
    if dtype not in (torch.float16,torch.bfloat16,torch.float32):raise UnsupportedCase('Inspected IEEE lookup covers FP16/BF16/FP32 only.')
    name=op_name(case);replacement=name=='NanToNumFwdOp';kind={'IsfiniteFwdOp':'isfinite','IsinfFwdOp':'isinf','IsnanFwdOp':'isnan'}.get(name)
    if replacement and (case.params.get('nan',0.0)!=0.0 or case.params.get('posinf') is not None or case.params.get('neginf') is not None):
        raise UnsupportedCase('Current immutable replacement table covers the complete default replacement manifest only.')
    binding=new_binding();key='float16' if dtype==torch.float16 else 'bfloat16'
    def run(x):
        raw=x.contiguous().reshape(-1).view(torch.uint8).reshape(-1,x.element_size())
        zero=torch.zeros_like(raw[:,0])
        high=torch.stack((raw[:,-2],raw[:,-1],zero,zero),dim=1).contiguous().view(torch.int32).reshape(-1)
        low=None if x.element_size()==2 else torch.stack((raw[:,0],raw[:,1],zero,zero),dim=1).contiguous().view(torch.int32).reshape(-1)
        axis=torch.tensor(0,device=x.device,dtype=torch.int32)
        # The following dictionaries exist for this invocation only. No
        # input-dependent indices, masks or output data survive a call.
        values={}
        def lookup(which,index=high):
            cache_key=(which,index is low)
            if cache_key not in values:
                dt=torch.int16 if which=='replace' else torch.uint8
                table=torch.tensor(table_values(key,which),device=x.device,dtype=dt)
                out=torch.empty(index.shape,device=x.device,dtype=dt)
                binding.call('GatherV2',[table,index,axis],[out],{'batch_dims':0,'negative_index_support':False},constants={2:(C.c_int32*1)(0)})
                values[cache_key]=out
            return values[cache_key]
        def multiply(a,b):
            out=torch.empty(a.shape,device=x.device,dtype=torch.float32)
            binding.call('ForeachMulList',[a.float(),b.float()],[out],{},names=['x10','x20','y0']);return out
        def classify(which):
            if low is None or which=='isfinite':return lookup(which)
            if which=='isinf':return multiply(lookup('isinf'),lookup('low_zero',low))
            # FP32's upper sixteen bits have BF16's exponent layout. Lower
            # mantissa bits turn an upper-half infinity pattern into NaN.
            extra=multiply(lookup('isinf'),lookup('low_nonzero',low));out=torch.empty_like(extra)
            binding.call('ForeachMaximumList',[lookup('isnan').float(),extra],[out],{},names=['x10','x20','y0']);return out
        if not replacement:return classify(kind).to(torch.bool).reshape(x.shape)
        if low is None:return lookup('replace').view(x.dtype).reshape(x.shape)
        masks=(classify('isnan'),multiply(lookup('positive_inf'),lookup('low_zero',low)),multiply(lookup('negative_inf'),lookup('low_zero',low)))
        result=x.contiguous().reshape(-1).clone()
        for mask,value in zip(masks,(0.0,torch.finfo(x.dtype).max,torch.finfo(x.dtype).min)):
            updates=torch.full(result.shape,value,device=x.device,dtype=x.dtype);out=torch.empty_like(result)
            binding.call('MaskedScatter',[result,mask.to(torch.bool),updates],[out],{});result=out
        return result.reshape(x.shape)
    api='GatherV2';dirs=['index/gather_v2']
    if dtype==torch.float32 and (replacement or kind in ('isinf','isnan')):
        api+=' + ForeachMulList';dirs.append('foreach/foreach_mul_list')
        if replacement or kind=='isnan':api+=' + ForeachMaximumList';dirs.append('foreach/foreach_maximum_list')
        if replacement:api+=' + MaskedScatter';dirs.append('index/masked_scatter')
    return candidate(binding,run,api,
                     'Lossless IEEE byte interleave forms bounded native lookup indices. Immutable exact exponent/mantissa tables feed original GatherV2; FP32 combines native 0/1 masks with original multiply/max, and default replacement uses original MaskedScatter. FP16/BF16 replacement selects complete raw result bits. All input transforms, device tables and masks are built within each call; no Torch predicate or replacement arithmetic.',*dirs)
