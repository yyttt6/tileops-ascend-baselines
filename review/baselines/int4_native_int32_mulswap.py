"""Full native Torch API composition; unchanged FP32 arithmetic and outputs."""
import torch
import torch_npu

def int4_group(w,group_size):
    n,k=w.shape
    groups=w.float().reshape(-1,group_size)
    lo,hi=groups.amin(-1),groups.amax(-1)
    multiplier=torch.where(hi==lo,1.0,16.0/(hi-lo))
    offset=(hi+lo)*0.5
    centered=groups-offset[:,None]
    # Native API packs first input into low nibble; workload packs it into high nibble.
    paired=centered
    packed32=torch_npu.npu_quantize(paired,multiplier,None,torch.quint4x2,-2,False)
    # Swap adjacent nibble order on packed INT32 words; full bitwise operation stays timed.
    mask=0x0F0F0F0F
    swapped=((packed32 & mask)*16)|((packed32>>4)&mask)
    packed=swapped.view(torch.int8).reshape(n,k//2)
    return packed,torch.stack((1.0/multiplier,offset),-1)
