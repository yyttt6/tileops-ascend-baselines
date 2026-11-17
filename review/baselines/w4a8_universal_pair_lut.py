"""Exact full-domain local decode table, independent of every model input."""
import torch

def cpu_pair_table():
    scale_codes=torch.arange(256,dtype=torch.uint8)
    signed=((torch.arange(16,dtype=torch.int32)^8)-8).float()
    # Exactly the original FP32 multiply then nearest-even FP8 weight rounding.
    values=(scale_codes.view(torch.float8_e4m3fn).float()[:,None]*signed).to(torch.float8_e4m3fn).view(torch.uint8)
    packed=torch.arange(256,dtype=torch.int64)
    pairs=torch.stack((values[:,packed&15],values[:,packed>>4]),-1).contiguous()
    return pairs.view(torch.int16).reshape(-1)

def decode(packed_weight,group_scale,group_size,pair_table):
    n=packed_weight.shape[0];k=packed_weight.shape[1]*2
    assert group_scale.dtype==torch.float8_e4m3fn
    packed=packed_weight.to(torch.int32).reshape(-1,group_size//2)
    # All input-dependent indexing and output materialization remain on device/timed.
    base=group_scale.view(torch.uint8).to(torch.int32).reshape(-1,1)*256
    index=(packed+base).reshape(-1)
    pairs=torch.index_select(pair_table,0,index)
    return pairs.view(torch.float8_e4m3fn).reshape(n,k)
