"""Original GatherV2 plus an immutable byte truth table for integer bitwise ops."""
import sys
from .common import candidate,new_binding,op_name


def dispatch(case):
    import torch
    if all(x.dtype==torch.bool for x in case.inputs):
        from .elementwise import boolean_logic
        return boolean_logic(case)
    return byte_lookup(case)


def byte_lookup(case):
    import ctypes as C
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    if sys.byteorder!='little':raise UnsupportedCase('Inspected lossless index layout requires little-endian storage.')
    name=op_name(case);binding=new_binding();dtype=getattr(torch,case.params['output'][1])
    if dtype not in (torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64):
        raise UnsupportedCase('Byte truth-table route covers declared integer result dtypes only.')
    # These 65,536 immutable values do not depend on any case input.
    if 'And' in name:data=bytes(a&b for a in range(256) for b in range(256))
    elif 'Or' in name:data=bytes(a|b for a in range(256) for b in range(256))
    else:data=bytes(a^b for a in range(256) for b in range(256))
    def run(left,right):
        shape=torch.broadcast_shapes(left.shape,right.shape)
        a=left.to(dtype).expand(shape).contiguous().reshape(-1).view(torch.uint8)
        b=right.to(dtype).expand(shape).contiguous().reshape(-1).view(torch.uint8)
        zero=torch.zeros_like(a)
        # Reinterpret [b, a, 0, 0] as INT32: exactly 256*a+b, with no
        # Torch arithmetic or narrowing of the original integer inputs.
        indices=torch.stack((b,a,zero,zero),dim=-1).contiguous().view(torch.int32).reshape(-1)
        table=torch.tensor(list(data),device=left.device,dtype=torch.uint8)
        axis=torch.tensor(0,device=left.device,dtype=torch.int32)
        out=torch.empty_like(a)
        binding.call('GatherV2',[table,indices,axis],[out],
                     {'batch_dims':0,'negative_index_support':False},constants={2:(C.c_int32*1)(0)})
        return out.view(dtype).reshape(shape)
    return candidate(binding,run,'GatherV2',
                     'Original GatherV2 selects immutable exact byte AND/OR/XOR truth-table entries. Per-call lossless integer promotion, broadcast and byte interleave form INT32 indices without input arithmetic; output bytes restore every integer bit. The lookup table contains only input-independent constants and its device copy is charged per call.',
                     'index/gather_v2')
